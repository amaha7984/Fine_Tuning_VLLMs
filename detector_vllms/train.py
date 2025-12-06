import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.explanation_only import ExplanationOnlyDataset
from vllms.transformer_backend import load_hf_qwen, format_qa_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="SFT on SynthCars explanation-only dataset")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/aul/homes/amaha038/DeepLearning/VLLMs/Fine_Tuning_VLLMs/detector_vllms/data/SynthScars/SynthScars/train",
        help="Path to SynthCars train folder (contains images/ and annotations/).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF model name or local path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/qwen_synthcars_expl",
        help="Where to save the fine-tuned model.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    return parser.parse_args()


def collate_fn(batch):
    """
    Collate:
      - stack image tensors (it is unused now)
      - keep questions and answers as lists of strings
    """
    images = torch.stack([b["image"] for b in batch])
    questions = [b["question"] for b in batch]
    answers = [b["answer"] for b in batch]
    return {
        "images": images,
        "questions": questions,
        "answers": answers,
    }


def main():
    args = parse_args()

    # 1. Dataset & DataLoader
    dataset = ExplanationOnlyDataset(root=args.data_root)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    print(f"Loaded dataset with {len(dataset)} samples from {args.data_root}")

    # 2. Model & tokenizer
    model, tokenizer = load_hf_qwen(
        model_name=args.model_name,
        torch_dtype=args.torch_dtype,
        device_map="auto",
    )

    model.train()
    # Optimizer (simple AdamW)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # For gradient accumulation
    global_step = 0

    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            questions = batch["questions"]
            answers = batch["answers"]

            # Build prompts: question + ground-truth explanation
            prompts = [
                format_qa_prompt(q, a)
                for q, a in zip(questions, answers)
            ]

            tokenized = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )

            # Move tokenized to model.device (works with device_map="auto")
            tokenized = {k: v.to(model.device) for k, v in tokenized.items()}

            # Standard causal LM loss: labels = input_ids
            outputs = model(**tokenized, labels=tokenized["input_ids"])
            loss = outputs.loss

            # Normalize by grad_accum_steps
            loss = loss / args.grad_accum_steps
            loss.backward()

            if (step + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                print(
                    f"Epoch {epoch} | Step {step} | Global step {global_step} | "
                    f"Loss: {loss.item() * args.grad_accum_steps:.4f}"
                )

        # (Optional) save checkpoint per epoch
        epoch_dir = Path(args.output_dir) / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        print(f"Saved model checkpoint to {epoch_dir}")

    # Final save
    out_dir = Path(args.output_dir) / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Training finished. Final model saved to {out_dir}")


if __name__ == "__main__":
    main()
