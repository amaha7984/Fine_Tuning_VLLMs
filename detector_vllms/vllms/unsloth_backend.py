from typing import Tuple

import torch
from unsloth import FastLanguageModel


def load_unsloth_qwen(
    model_name: str = "unsloth/Qwen2.5-7B-Instruct",
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    lora_r: int = 16,
) -> Tuple[torch.nn.Module, "PreTrainedTokenizer"]:
    """
    Load Qwen with Unsloth, ready for LoRA finetuning.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        
    )

    # Attach LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules="all-linear",       
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    return model, tokenizer

#dataset preparation is pending
def format_qa_prompt(question: str, answer: str | None = None) -> str:
   
    if answer is None:
        return f"User: {question}\nAssistant:"
    return f"User: {question}\nAssistant: {answer}"