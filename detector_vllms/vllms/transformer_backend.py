from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
import torch


def load_hf_qwen_vl(
    model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    dtype="bfloat16",
    device_map="auto",
):
    """Load Qwen2.5-VL correctly as a multimodal model."""

    if torch.cuda.is_available():
        torch_dtype = getattr(torch, dtype)
    else:
        torch_dtype = torch.float32
        device_map = None

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Ensure tokenizer has a pad token
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch_dtype,        # NOTE: use dtype= not torch_dtype=
        device_map=device_map,
        trust_remote_code=True,
    )

    return model, processor
