# train_qlora.py
# QLoRA fine-tuning for Qwen2.5-VL and Mistral3-VL 

import argparse
from pathlib import Path
import os

import torch
from torch.utils.data import DataLoader

from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

from datasets.explanation_only import ExplanationOnlyVLDataset
from vllms.transformer_backend_qlora import (
    load_qwen2_5_vl,
    load_mistral3_vl,
)


def parse_args():
    p = argparse.ArgumentParser(description="QLoRA fine-tune VLMs on SynthScars explanation-only dataset")

    p.add_argument(
        "--data_root",
        type=str,
        default="/aul/homes/amaha038/DeepLearning/VLLMs/Fine_Tuning_VLLMs/detector_vllms/data/SynthScars/SynthScars/train",
        help="Path to train folder (contains images/ and annotations/).",
    )
    p.add_argument(
        "--model_family",
        type=str,
        choices=["qwen2_5_vl", "mistral3_vl"],
        default="qwen2_5_vl",
    )
    p.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="HF model name or local path.",
    )
    p.add_argument("--output_dir", type=str, default="./outputs_qlora")

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--max_length", type=int, default=1024)

    # LoRA knobs
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_targets",
        type=str,
        default="auto",
        help="Comma-separated target module suffixes, or 'auto'.",
    )

    # QLoRA knobs
    p.add_argument(
        "--compute_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Compute dtype for 4-bit quantized model.",
    )

    # Stability / memory
    p.add_argument(
        "--gradient_checkpointing",
        type=int,
        default=1,
        help="1 to enable gradient checkpointing (recommended for 40GB).",
    )

    return p.parse_args()


def collate_fn_vl(batch):
    return {
        "images": [b["image"] for b in batch],          # list[PIL.Image]
        "questions": [b["question"] for b in batch],    # list[str]
        "answers": [b["answer"] for b in batch],        # list[str]
        "image_paths": [b["image_path"] for b in batch],
    }


def build_conversations(questions, answers):
    conversations = []
    for q, a in zip(questions, answers):
        conversations.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": (
                                "You are an expert forensic analyst for AI-generated images. "
                                "Analyze the provided image and answer the following question.\n\n"
                                f"Question: {q}"
                            ),
                        },
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": a}]},
            ]
        )
    return conversations


def _split_csv(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def infer_lora_target_modules(model) -> list[str]:
    candidates = {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "out_proj", "fc1", "fc2",
    }
    found = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            suffix = name.split(".")[-1]
            if suffix in candidates:
                found.add(suffix)

    preferred = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    ordered = [x for x in preferred if x in found]
    return ordered if ordered else (sorted(found) if found else ["q_proj", "k_proj", "v_proj", "o_proj"])


def apply_qlora(model, args):
    # 1) Prepare k-bit training (casts norms, sets up grads correctly)
    model = prepare_model_for_kbit_training(model)

    # 2) Enable gradient checkpointing
    if args.gradient_checkpointing == 1 and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Disabling cache for training
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # 3) Decide target modules
    if args.lora_targets.lower() == "auto":
        target_modules = infer_lora_target_modules(model)
    else:
        target_modules = _split_csv(args.lora_targets)

    print(f"[QLoRA] Target modules: {target_modules}")
    print(f"[QLoRA] r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")

    # 4) Inject LoRA adapters
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def cast_vision_tower_dtype(model, compute_dtype: torch.dtype):
    """
    Fix dtype mismatch between pixel_values (often bf16) and vision conv weights (sometimes fp32).
    """
    for attr in ["vision_tower", "visual", "vision_model"]:
        if hasattr(model, attr):
            try:
                getattr(model, attr).to(dtype=compute_dtype)
                print(f"[QLoRA] Casted {attr} to {compute_dtype}")
                return
            except Exception as e:
                print(f"[QLoRA] Could not cast {attr}: {e}")
                return
    # Some wrappers place the actual model under `model` or `base_model`
    for attr in ["model", "base_model"]:
        if hasattr(model, attr):
            sub = getattr(model, attr)
            for vattr in ["vision_tower", "visual", "vision_model"]:
                if hasattr(sub, vattr):
                    try:
                        getattr(sub, vattr).to(dtype=compute_dtype)
                        print(f"[QLoRA] Casted {attr}.{vattr} to {compute_dtype}")
                        return
                    except Exception as e:
                        print(f"[QLoRA] Could not cast {attr}.{vattr}: {e}")
                        return


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    compute_dtype = getattr(torch, args.compute_dtype)

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    device_map = {"": 0}

    dataset = ExplanationOnlyVLDataset(root=args.data_root)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_vl,
    )
    print(f"[VLM] Loaded dataset with {len(dataset)} samples from {args.data_root}")
    print(f"[VLM] Using model_family={args.model_family}, model_name={args.model_name}")

    # Load model + processor in 4-bit
    if args.model_family == "qwen2_5_vl":
        model, processor = load_qwen2_5_vl(
            model_name=args.model_name,
            dtype=args.compute_dtype,
            device_map=device_map,
            quantization_config=bnb_cfg,
        )
    elif args.model_family == "mistral3_vl":
        model, processor = load_mistral3_vl(
            model_name=args.model_name,
            dtype=args.compute_dtype,
            device_map=device_map,
            quantization_config=bnb_cfg,
        )
    else:
        raise ValueError(f"Unknown model_family: {args.model_family}")

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Processor has no `tokenizer` attribute.")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    cast_vision_tower_dtype(model, compute_dtype)

    # Apply QLoRA (prepare k-bit + LoRA injection)
    model = apply_qlora(model, args)
    model.train()

    device = next(model.parameters()).device

    # Optimizer: ONLY trainable params (LoRA)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters found. QLoRA injection likely failed.")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    global_step = 0

    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            images = batch["images"]
            questions = batch["questions"]
            answers = batch["answers"]

            conversations = build_conversations(questions, answers)

            # Chat templating
            if args.model_family == "qwen2_5_vl":
                prompts = tokenizer.apply_chat_template(
                    conversations,
                    add_generation_prompt=False,
                    tokenize=False,
                )
            else:  # mistral3_vl
                prompts = processor.apply_chat_template(
                    conversations,
                    add_generation_prompt=False,
                    tokenize=False,
                )

            if isinstance(prompts, str):
                prompts = [prompts]

            inputs = processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )

            # Move tensors to GPU
            inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

            input_ids = inputs["input_ids"]
            labels = input_ids.clone()
            if tokenizer.pad_token_id is not None:
                labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            loss = loss / args.grad_accum_steps
            loss.backward()

            if (step + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                print(
                    f"[QLoRA] Epoch {epoch} | Step {step} | Global {global_step} | "
                    f"Loss: {loss.item() * args.grad_accum_steps:.4f}"
                )

        # Save adapters per epoch
        epoch_dir = Path(args.output_dir) / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(epoch_dir)      # saves adapters (PEFT)
        processor.save_pretrained(epoch_dir)
        print(f"[QLoRA] Saved epoch {epoch} adapters to {epoch_dir}")

    # Final save
    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"[QLoRA] Training finished. Final adapters saved to {final_dir}")


if __name__ == "__main__":
    main()
