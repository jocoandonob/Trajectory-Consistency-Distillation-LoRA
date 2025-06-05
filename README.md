---
title: Tcdl
emoji: 📚
colorFrom: pink
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# TCD LoRA Demo
![Thumbnail](thumbnail.png)

This is a Gradio demo for the TCD LoRA model, deployed on Hugging Face Spaces.

## Configuration

This Space is configured with:
- Python 3.10
- Gradio 4.0.0
- PyTorch 2.0.0
- Diffusers 0.24.0

## Dependencies

The following dependencies are required:
```bash
torch>=2.0.0
diffusers>=0.24.0
transformers>=4.36.0
accelerate>=0.25.0
gradio>=4.0.0
safetensors>=0.4.0 
peft>=0.7.0
requests>=2.31.0
numpy>=1.24.0
Pillow>=10.0.0
```

## Usage

1. Enter your prompt in the text box
2. Adjust the generation parameters:
   - Seed: Controls the randomness of the generation
   - Number of Steps: More steps = better quality but slower
   - Guidance Scale: Higher values = more prompt adherence
   - Eta: Controls the noise level
3. Click "Generate Image" to create your image

## Model Information

This demo uses:
- Base model: Stable Diffusion XL
- LoRA: TCD-SDXL-LoRA
