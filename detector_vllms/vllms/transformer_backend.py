from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_hf_qwen(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype: str = "bfloat16",
    device_map: str | None = "auto",
) -> Tuple[torch.nn.Module, "AutoTokenizer"]:
    """
    Load Qwen via Hugging Face transformers.

    This uses a causal LM (text-only)
    """
    if torch.cuda.is_available():
        dtype = getattr(torch, torch_dtype)
    else:
        dtype = torch.float32
        device_map = None  # CPU only

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    return model, tokenizer


def format_qa_prompt(question: str, answer: str | None = None) -> str:
    """
    Simple instruction-style prompt.
    """
    if answer is None:
        return (
            "You are an AI-forensics expert. Analyze the image and answer the user.\n\n"
            f"User: {question}\nAssistant:"
        )
    return (
        "You are an AI-forensics expert. Analyze the image and answer the user.\n\n"
        f"User: {question}\nAssistant: {answer}"
    )