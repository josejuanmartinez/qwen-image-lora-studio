---
title: Qwen Image LoRA Studio
emoji: 🎨
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.26.0
python_version: 3.11
app_file: app.py
pinned: false
---

# Private Qwen Image LoRA Studio

Train private Qwen-Image-2512 LoRAs with [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit), then run them through the UI or `POST /v1/generate`.

The Train tab keeps uploaded images in a paginated thumbnail gallery. Common LoRA, optimizer, dataset, sampling, quantization, and checkpoint settings are available in collapsed advanced sections. The final **All AI Toolkit parameters** section accepts optional YAML that is merged into the generated `sd_trainer` configuration, for example:

```yaml
train:
  optimizer_params:
    weight_decay: 0.01
  noise_offset: 0.05
network:
  network_kwargs:
    only_if_contains: [attn]
```

The Space intentionally uses Python 3.11 because AI Toolkit currently pins SciPy 1.12, which does not provide Python 3.13 wheels.
AI Toolkit is installed in an isolated virtual environment so its pinned dependencies cannot modify the running Gradio application. The Train tab streams installation and trainer output into a fixed, scrollable live console and prevents another training submission while a job is active. Job status and bounded logs are also saved under `/tmp/qwen-image-lora-studio/jobs/<job-name>/`, so refreshing Gradio automatically reopens the latest job and continues displaying its log.
The toolkit environment also pins `kernels==0.12.3`, matching the API required by its Transformers 5.5.x dependency. For PyTorch 2.11 and newer it installs TorchAudio 2.11, which uses PyTorch's stable ABI; older PyTorch releases receive their matching TorchAudio build.

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
  "seed": 42,
  "scheduler": null,
  "base_model": null,
  "remove_background": true,
  "upscale_to_2k": false
}
```

The response contains a PNG as base64 plus its dimensions, transparency status, upscaler details, used seed, and a replayable `generation_parameters` object. Reuse those parameter fields in the approved-final request so prompt, LoRA, negative prompt, steps, guidance, LoRA scale, seed, scheduler, and base model remain locked. The draft and final both generate at 1024×1024; only the final sets `upscale_to_2k: true`, producing 2048×2048. When `remove_background` is enabled, `rembg[gpu]` runs through ONNX Runtime's CUDA execution provider and returns a genuine alpha PNG. If Swin2SR cannot load or fit in memory, the response reports that it used the Lanczos fallback. LoRA names resolve to `jjmcarrascosa/<lora_name>`; fully-qualified private model IDs also work.

## Hardware

This application requires a paid GPU. For training, use at least a 24 GB GPU; 48 GB+ is recommended for Qwen Image 2512. The first inference loads roughly 20B model parameters and can take several minutes.
