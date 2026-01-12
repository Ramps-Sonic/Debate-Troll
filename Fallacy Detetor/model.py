import warnings
from typing import Optional

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
)
from peft import prepare_model_for_kbit_training

# Optional: 4-bit quantization via bitsandbytes
try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except ImportError:
    BitsAndBytesConfig = None  # type: ignore
    _HAS_BNB = False

from peft import LoraConfig, get_peft_model

# Import label definitions from dataset.py
from dataset import FALLACY_LABELS


def print_trainable_parameters(model) -> None:
    """
    Utility to log how many parameters are trainable.
    Useful to verify that LoRA is correctly applied.
    """
    trainable_params = 0
    all_params = 0
    for _, p in model.named_parameters():
        num_params = p.numel()
        all_params += num_params
        if p.requires_grad:
            trainable_params += num_params
    pct = 100 * trainable_params / all_params
    print(
        f"Trainable params: {trainable_params} | All params: {all_params} | "
        f"Trainable%: {pct:.4f}"
    )


def create_fallacy_model(
    model_name: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[list] = None,
    use_4bit: bool = True,
    device_map: Optional[str] = "auto",
):
    """
    Create a sequence classification model for logical fallacy detection
    with LoRA (QLoRA-style) on top of a base LLM.

    Args:
        model_name:
            HuggingFace model name or local path, e.g. "meta-llama/Meta-Llama-3-8B".
        lora_r:
            Rank of the LoRA matrices.
        lora_alpha:
            Scaling factor for LoRA.
        lora_dropout:
            Dropout for LoRA layers.
        target_modules:
            List of module names to apply LoRA on.
            If None, a reasonable default will be used for common transformer models
            like "q_proj", "v_proj", etc.
        use_4bit:
            Whether to load the base model in 4-bit quantized mode (QLoRA style).
            Requires bitsandbytes installed.
        device_map:
            Device map for model loading. "auto" is usually fine.

    Returns:
        model: A PEFT-wrapped model ready for training or inference.
    """
    num_labels = len(FALLACY_LABELS)

    # ----------------------------------------------------------------------
    # Configure quantization (4-bit) if requested and bitsandbytes is available
    # ----------------------------------------------------------------------
    quantization_config = None
    if use_4bit:
        if not _HAS_BNB:
            raise ImportError(
                "BitsAndBytesConfig not available. "
                "Install bitsandbytes or set use_4bit=False."
            )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # ----------------------------------------------------------------------
    # Base config and model
    # ----------------------------------------------------------------------
    model_config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=model_config,
        quantization_config=quantization_config,
        device_map=device_map,
    )

    # ----------------------------------------------------------------------
    # LoRA configuration
    # ----------------------------------------------------------------------
    if target_modules is None:
        # A common default for many transformer-based LLMs
        # You may adjust this list based on your backbone.
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        warnings.warn(
            "No target_modules provided for LoRA. "
            "Using default: ['q_proj', 'k_proj', 'v_proj', 'o_proj', "
            "'gate_proj', 'up_proj', 'down_proj']. "
            "Please verify they match your model architecture."
        )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=target_modules,
    )
    # 加载量化模型之后
    model.config.use_cache = False  # 和 checkpointing 一致，避免隐患

    # 关键：QLoRA 稳定性步骤
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    model = get_peft_model(model, lora_config)

    # Optional: show how many parameters are trainable
    print_trainable_parameters(model)

    return model
