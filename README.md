---
title: Qwen Image LoRA Studio
emoji: 🎨
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
---

# Private Qwen Image LoRA Studio

Train private Qwen-Image-2512 LoRAs with [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit), then run them through the UI or `POST /v1/generate`.

## Space secrets

Set `HF_TOKEN` in **Settings → Variables and secrets**. It must have write access to create and upload private model repositories under this account.

## API

`POST /v1/generate`

```json
{
  "prompt": "portrait photo of mysubject in a forest",
  "lora_name": "mysubject-v1",
  "negative_prompt": "blurry",
  "width": 1024,
  "height": 1024,
  "steps": 28,
  "guidance_scale": 4.0,
  "lora_scale": 0.8,
  "seed": 42
}
```

The response contains a PNG as base64 plus the used seed. LoRA names resolve to `jjmcarrascosa/<lora_name>`; fully-qualified private model IDs also work.

## Hardware

This application requires a paid GPU. For training, use at least a 24 GB GPU; 48 GB+ is recommended for Qwen Image 2512. The first inference loads roughly 20B model parameters and can take several minutes.