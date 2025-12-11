# train.py
import argparse
from pathlib import Path
import os

import torch
from torch.utils.data import DataLoader

from datasets.explanation_only import ExplanationOnlyVLDataset
from vllms.transformer_backend import (
    load_qwen2_5_vl,
    load_qwen3_vl,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune VLMs/MLLMs on SynthScars explanation-only dataset"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/aul/homes/amaha038/DeepLearning/VLLMs/Fine_Tuning_VLLMs/detector_vllms/data/SynthScars/SynthScars/train",
        help="Path to SynthScars train folder (contains images/ and annotations/).",
    )

    # NEW: choose which family of model to use
    parser.add_argument(
        "--model_family",
        type=str,
        choices=["qwen2_5_vl", "qwen3_vl"],
        default="qwen2_5_vl",
        help="Which Qwen VLM family to use.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help=(
            "Hugging Face model name or local path. "
            "For qwen2_5_vl: e.g., Qwen/Qwen2.5-VL-3B-Instruct "
            "For qwen3_vl: e.g., Qwen/Qwen3-VL-4B-Instruct"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/qwen2_5_vl_synthscars_expl",
        help="Directory to save fine-tuned model checkpoints.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Model dtype for CUDA (e.g., bfloat16, float16, float32).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Dataloader workers.",
    )
    return parser.parse_args()


def collate_fn_vl(batch):
    """
    Collate function for VLM training.

    Each item from ExplanationOnlyVLDataset is:
      {
        "image": PIL.Image,
        "image_path": str,
        "question": str,
        "answer": str,
      }
    """
    images = [b["image"] for b in batch]         # list[PIL.Image]
    questions = [b["question"] for b in batch]   # list[str]
    answers = [b["answer"] for b in batch]       # list[str]
    image_paths = [b["image_path"] for b in batch]
    return {
        "images": images,
        "questions": questions,
        "answers": answers,
        "image_paths": image_paths,
    }


def build_conversations(questions, answers):
    """
    Build chat-format conversations for Qwen2.5-VL and Qwen3-VL.

    For each sample:
      user: [image + question]
      assistant: explanation (caption)

    NOTE: content MUST always be a list of dicts (multimodal format).
    """
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
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": a,
                        }
                    ],
                },
            ]
        )
    return conversations


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Dataset & DataLoader
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

    # 2. Load model + processor based on model_family
    if args.model_family == "qwen2_5_vl":
        model, processor = load_qwen2_5_vl(
            model_name=args.model_name,
            dtype=args.dtype,
            device_map="auto",
        )
    elif args.model_family == "qwen3_vl":
        model, processor = load_qwen3_vl(
            model_name=args.model_name,
            dtype=args.dtype,
            device_map="auto",
        )
    else:
        raise ValueError(f"Unknown model_family: {args.model_family}")

    # For device handling: with device_map='auto', parameters are on CUDA shards.
    # Use the first parameter's device as the "main" device for tensors.
    device = next(model.parameters()).device
    model.train()

    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    global_step = 0

    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            images = batch["images"]          # list of PIL images
            questions = batch["questions"]
            answers = batch["answers"]

            # 3.1 Build chat-format conversations
            conversations = build_conversations(questions, answers)

            # 3.2 Use *tokenizer*'s chat template to get text with vision tokens
            #     We keep tokenize=False so we get raw strings.
            prompts = tokenizer.apply_chat_template(
                conversations,
                add_generation_prompt=False,  # we include assistant messages already
                tokenize=False,
            )

            # HF returns a string for single conversation, list for multiple
            if isinstance(prompts, str):
                prompts = [prompts]

            # 3.3 Now call the processor with text + images
            inputs = processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=False,   # IMPORTANT: don't truncate multimodal sequences
                # max_length=args.max_length,  # not used here to avoid mm-token mismatch
            )

            # Move tensors to model device
            inputs = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in inputs.items()
            }

            # 3.4 Labels: mask padding tokens
            input_ids = inputs["input_ids"]
            labels = input_ids.clone()
            if tokenizer.pad_token_id is not None:
                labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            # Gradient accumulation
            loss = loss / args.grad_accum_steps
            loss.backward()

            if (step + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                print(
                    f"[VLM] Epoch {epoch} | Step {step} | Global step {global_step} | "
                    f"Loss: {loss.item() * args.grad_accum_steps:.4f}"
                )

        # 4. Save checkpoint per epoch
        epoch_dir = Path(args.output_dir) / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(epoch_dir)
        processor.save_pretrained(epoch_dir)
        print(f"[VLM] Saved epoch {epoch} checkpoint to {epoch_dir}")

    # 5. Final save
    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"[VLM] Training finished. Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
