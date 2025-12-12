#transformer_backend.py
import torch
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForImageTextToText,
   # Qwen3VLForConditionalGeneration,
)

def _resolve_dtype_and_device(dtype: str, device_map: str | None):
    """
    Helper to map string dtype -> torch dtype and handle CPU fallback.
    """
    if torch.cuda.is_available():
        try:
            torch_dtype = getattr(torch, dtype)
        except AttributeError:
            print(f"[backend] Unknown dtype '{dtype}', falling back to bfloat16.")
            torch_dtype = torch.bfloat16
        # keep device_map as passed (e.g., "auto")
        return torch_dtype, device_map
    else:
        # On CPU: always float32, no device_map sharding
        print("[backend] CUDA not available, using float32 on CPU.")
        return torch.float32, None

# -------------------------------------------------------------------------
# Qwen2.5-VL loader
# -------------------------------------------------------------------------
def load_qwen2_5_vl(
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    dtype: str = "bfloat16",
    device_map: str | None = "auto",
):
    """Load Qwen2.5-VL correctly as a multimodal model."""

    torch_dtype, device_map = _resolve_dtype_and_device(dtype, device_map)

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Ensure tokenizer has a pad token
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch_dtype,          # NOTE: use dtype= (HF Qwen2.5-VL convention)
        device_map=device_map,
        trust_remote_code=True,
    )

    model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    return model, processor


# -------------------------------------------------------------------------
# Qwen3-VL loader
# -------------------------------------------------------------------------
"""
def load_qwen3_vl(
    model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
    dtype: str = "bfloat16",
    device_map: str | None = "auto",
):

    torch_dtype, device_map = _resolve_dtype_and_device(dtype, device_map)

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch_dtype,              # dtype, consistent with docs
        device_map=device_map,
        attn_implementation="sdpa",     
        trust_remote_code=True,
    )

    model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    return model, processor
"""

# -------------------------------------------------------------------------
# Mistral 3 vision-language loader 
# -------------------------------------------------------------------------
def load_mistral3_vl(
    model_name: str = "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    dtype: str = "bfloat16",
    device_map: str | None = "auto",
):
    """
    Load Mistral 3 image-text-to-text model.

    Uses AutoModelForImageTextToText + AutoProcessor as in HF docs.
    """

    torch_dtype, device_map = _resolve_dtype_and_device(dtype, device_map)

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if hasattr(processor, "tokenizer") and processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # For long context + memory friendliness
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    return model, processor