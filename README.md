# Fine_Tuning_VLLMs

Finetuning Vision-Language Models (VLMs) and Vision Large Language Models (VLLMs) for AI-generated content detection.

Supported models:
- Qwen3-VL (Qwen2.5)
- LLaMA 3.2 Vision
- Mistral VLM
- Gemma 3 Vision

## LoRA Fine-Tuning
a) Description
- LoRA fine-tuning adapts a pretrained model without changing its original weights.
- The original weights remain frozen during training. Instead, LoRA learns the weight changes.
- These changes are represented using two small trainable matrices.
- The matrices are multiplied to form a weight update with the same size as the original weight matrix.
- Only these low-rank matrices are trained, which reduces memory and computation cost.

## QLoRA Fine-Tuning
a) Description:
- QLoRA fine-tuning extends LoRA by applying quantization to the pretrained model weights.
- The original model weights are quantized to low-bit precision and kept frozen during training.
- LoRA is then applied on top of the quantized weights using low-rank trainable matrices.
- Only the low-rank matrices are trained, while the quantized base model remains fixed.
- This approach significantly reduces GPU memory usage and enables fine-tuning large models on limited hardware.
