# Fine_Tuning_VLLMs

Finetuning Vision-Language Models (VLMs) and Vision Large Language Models (VLLMs) for AI-generated content detection.

Supported models:
- Qwen3-VL (Qwen2.5)
- LLaMA 3.2 Vision
- Mistral VLM
- Gemma 3 Vision

## QLoRA Fine-Tuning
a) Description:
- QLoRA fine-tuning extends LoRA by applying quantization to the pretrained model weights.
- The original model weights are quantized to low-bit precision and kept frozen during training.
- LoRA is then applied on top of the quantized weights using low-rank trainable matrices.
- Only the low-rank matrices are trained, while the quantized base model remains fixed.
- This approach significantly reduces GPU memory usage and enables fine-tuning large models on limited hardware.
