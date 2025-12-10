import argparse
from pathlib import Path
import os
import json

import torch
from torch.utils.data import DataLoader

from datasets.explanation_only import ExplanationOnlyVLDataset
from vllms.transformer_backend import load_hf_qwen_vl  # we will only use processor/tokenizer logic


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned Qwen2.5-VL on SynthScars test set"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/aul/homes/amaha038/DeepLearning/VLLMs/Fine_Tuning_VLLMs/detector_vllms/data/SynthScars/SynthScars/test",
        help="Path to SynthScars *test* folder (contains images/ and annotations/).",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./outputs/qwen2_5_vl_synthscars_expl/final",
        help="Path to fine-tuned model directory (the 'final' or specific epoch dir).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Evaluation batch size (keep small for VLM).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate per sample.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Dataloader workers.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./outputs/qwen2_5_vl_synthscars_expl/test_predictions.jsonl",
        help="Where to save JSONL with image path, GT caption, and generated explanation.",
    )
    parser.add_argument(
        "--print_examples",
        type=int,
        default=5,
        help="Number of examples to print to stdout for inspection.",
    )
    return parser.parse_args()


def collate_fn_vl(batch):
    """Same structure as in train.py."""
    images = [b["image"] for b in batch]
    questions = [b["question"] for b in batch]
    answers = [b["answer"] for b in batch]
    image_paths = [b["image_path"] for b in batch]
    return {
        "images": images,
        "questions": questions,
        "answers": answers,
        "image_paths": image_paths,
    }


def build_eval_conversations(questions):
    """
    Build chat-format conversations for *inference*.

    Only user message:
      - image
      - question text

    We let the model generate the assistant's explanation.
    """
    conversations = []
    for q in questions:
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
                }
            ]
        )
    return conversations


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # 1. Dataset & DataLoader (test split)
    dataset = ExplanationOnlyVLDataset(root=args.data_root, split="test")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_vl,
    )

    print(f"[VLM-EVAL] Loaded TEST dataset with {len(dataset)} samples from {args.data_root}")

    # 2. Load fine-tuned model + processor
    #    We don't need dtype/device_map here; AutoModel will infer device from .to(device)
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
    )
    tokenizer = processor.tokenizer

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Iterate over test set, generate explanations
    results = []
    example_printed = 0

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            images = batch["images"]
            questions = batch["questions"]
            answers = batch["answers"]          # ground-truth captions
            image_paths = batch["image_paths"]

            # 3.1 Build user-only conversations (no assistant content)
            conversations = build_eval_conversations(questions)

            # 3.2 Build text prompts with tokenizer chat template
            prompts = tokenizer.apply_chat_template(
                conversations,
                add_generation_prompt=True,  # important: we want the model to continue as assistant
                tokenize=False,
            )
            if isinstance(prompts, str):
                prompts = [prompts]

            # 3.3 Processor to handle text + images
            inputs = processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )

            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            prompt_len = inputs["input_ids"].shape[1]


            # 3.4 Generate
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,       
            )

            
            generated_texts = []
            for g in generated_ids:
                new_tokens = g[prompt_len:]
                text = tokenizer.decode(
                    new_tokens,
                    skip_special_tokens=True,
                ).strip()
                generated_texts.append(text)

            # Store results
            for img_path, q, gt, gen in zip(image_paths, questions, answers, generated_texts):
                results.append(
                    {
                        "image_path": img_path,
                        "question": q,
                        "ground_truth_explanation": gt,
                        "generated_explanation": gen,
                    }
                )

                # Print a few examples
                if example_printed < args.print_examples:
                    print("=" * 80)
                    print(f"[Example {example_printed}] image: {img_path}")
                    print(f"Q: {q}")
                    print(f"GT: {gt}")
                    print(f"PRED: {gen}")
                    example_printed += 1

    # 4. Save all predictions to JSONL
    save_path = Path(args.save_path)
    with save_path.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[VLM-EVAL] Saved {len(results)} predictions to {save_path}")


if __name__ == "__main__":
    main()
