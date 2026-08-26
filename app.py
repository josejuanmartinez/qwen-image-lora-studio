import base64
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import gradio as gr
import torch
import yaml
from diffusers import DiffusionPipeline
from fastapi import HTTPException
from huggingface_hub import HfApi
from pydantic import BaseModel, Field

try:
    import spaces
except ImportError:  # Allows local development outside Hugging Face Spaces.
    class _SpacesFallback:
        @staticmethod
        def GPU(*_args, **_kwargs):
            return lambda function: function
    spaces = _SpacesFallback()

BASE_MODEL = "Qwen/Qwen-Image-2512"
OWNER = os.getenv("HF_SPACE_OWNER", "jjmcarrascosa")
ROOT = Path(os.getenv("SPACE_WORKDIR", "/tmp/qwen-image-lora-studio"))
TOOLKIT = ROOT / "ai-toolkit"
JOBS = ROOT / "jobs"
JOBS.mkdir(parents=True, exist_ok=True)

pipe = None
loaded_lora = None
pipe_lock = threading.Lock()
jobs: dict[str, dict] = {}


def hf_token() -> str:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("Set an HF_TOKEN Space secret with write access before training or using private LoRAs.")
    return token


def slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-.").lower()
    if not value:
        raise ValueError("A LoRA name is required.")
    return value[:96]


def lora_repo(name: str) -> str:
    return name if "/" in name else f"{OWNER}/{slug(name)}"


def ensure_toolkit() -> None:
    if not TOOLKIT.exists():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ostris/ai-toolkit.git", str(TOOLKIT)], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=TOOLKIT, check=True)


def qwen_config(run_name: str, image_dir: Path, trigger_word: str, steps: int) -> dict:
    return {
        "job": "extension",
        "config": {
            "name": run_name,
            "process": [{"type": "sd_trainer"}],
            "training_folder": str(JOBS),
            "device": "cuda:0",
            "trigger_word": trigger_word,
            "network": {"type": "lora", "linear": 16, "linear_alpha": 16},
            "save": {"dtype": "float16", "save_every": max(100, steps // 4), "max_step_saves_to_keep": 2},
            "datasets": [{"folder_path": str(image_dir), "caption_ext": "txt", "caption_dropout_rate": 0.05, "shuffle_tokens": False, "cache_latents_to_disk": True, "resolution": [512, 768, 1024]}],
            "train": {"batch_size": 1, "cache_text_embeddings": True, "steps": steps, "gradient_accumulation": 1, "train_unet": True, "train_text_encoder": False, "gradient_checkpointing": True, "noise_scheduler": "flowmatch", "optimizer": "adamw8bit", "lr": 0.0001, "dtype": "bf16"},
            "model": {"name_or_path": BASE_MODEL, "arch": "qwen_image", "quantize": True, "qtype": "uint4|ostris/accuracy_recovery_adapters/qwen_image_2512_torchao_uint4.safetensors", "quantize_te": True, "qtype_te": "qfloat8", "low_vram": True},
            "sample": {"sampler": "flowmatch", "sample_every": max(100, steps // 4), "sample_start_step": 0, "width": 1024, "height": 1024, "prompts": [f"{trigger_word}, portrait photo, detailed"], "neg": "blurry, low quality", "seed": 42, "walk_seed": False, "guidance_scale": 4, "sample_steps": 28},
            "meta": {"name": run_name, "base_model": BASE_MODEL, "trigger_word": trigger_word, "private": True},
        },
    }


def run_training(run_name: str, image_dir: Path, trigger_word: str, steps: int) -> None:
    job = jobs[run_name]
    try:
        hf_token()
        ensure_toolkit()
        config_file = JOBS / f"{run_name}.yaml"
        config_file.write_text(yaml.safe_dump(qwen_config(run_name, image_dir, trigger_word, steps), sort_keys=False), encoding="utf-8")
        job["status"] = "training"
        result = subprocess.run([sys.executable, "run.py", str(config_file)], cwd=TOOLKIT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        job["log"] = result.stdout[-12000:]
        if result.returncode:
            raise RuntimeError(f"AI Toolkit exited with code {result.returncode}")
        weights = sorted((JOBS / run_name).rglob("*.safetensors"), key=lambda p: p.stat().st_mtime)
        if not weights:
            raise RuntimeError("AI Toolkit completed but did not produce a .safetensors LoRA.")
        api = HfApi(token=hf_token())
        repo_id = lora_repo(run_name)
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=str(weights[-1]), path_in_repo="adapter.safetensors", repo_id=repo_id, repo_type="model", commit_message=f"Publish private Qwen Image LoRA {run_name}")
        api.upload_file(path_or_fileobj=str(config_file), path_in_repo="training_config.yaml", repo_id=repo_id, repo_type="model", commit_message="Add training configuration")
        job.update(status="published", repo_id=repo_id, log=job.get("log", "") + f"\nPublished privately to {repo_id}")
    except Exception as exc:
        job.update(status="failed", log=job.get("log", "") + f"\nERROR: {exc}")


def start_training(files, lora_name: str, trigger_word: str, caption: str, steps: int):
    run_name = slug(lora_name)
    if not files:
        raise gr.Error("Upload one or more training images.")
    if not trigger_word.strip():
        raise gr.Error("Provide a trigger word, such as mysubject.")
    image_dir = JOBS / run_name / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(files):
        source = Path(path)
        target = image_dir / f"{index:04d}{source.suffix.lower()}"
        shutil.copy2(source, target)
        target.with_suffix(".txt").write_text(caption.strip() or trigger_word.strip(), encoding="utf-8")
    jobs[run_name] = {"status": "queued", "repo_id": lora_repo(run_name), "log": "Queued"}
    threading.Thread(target=run_training, args=(run_name, image_dir, trigger_word.strip(), int(steps)), daemon=True).start()
    return f"Queued `{run_name}`. It will publish privately to `{lora_repo(run_name)}` when complete."


def status(lora_name: str) -> str:
    job = jobs.get(slug(lora_name))
    return "No local job found." if job is None else yaml.safe_dump(job, sort_keys=False)


def get_pipe(lora_name: Optional[str], scale: float):
    global pipe, loaded_lora
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU Space is required for Qwen Image inference.")
    with pipe_lock:
        if pipe is None:
            pipe = DiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, token=hf_token()).to("cuda")
        repo = lora_repo(lora_name) if lora_name else None
        if repo != loaded_lora:
            if loaded_lora:
                pipe.unload_lora_weights()
            if repo:
                pipe.load_lora_weights(repo, weight_name="adapter.safetensors", token=hf_token(), adapter_name="selected")
                pipe.set_adapters("selected", adapter_weights=scale)
            loaded_lora = repo
        elif repo:
            pipe.set_adapters("selected", adapter_weights=scale)
        return pipe


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    lora_name: Optional[str] = None
    negative_prompt: str = ""
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    steps: int = Field(default=28, ge=1, le=80)
    guidance_scale: float = Field(default=4.0, ge=0, le=20)
    lora_scale: float = Field(default=0.8, ge=0, le=2)
    seed: Optional[int] = None


def generate(request: GenerateRequest):
    try:
        seed = int(request.seed if request.seed is not None else torch.seed() % (2**31 - 1))
        generator = torch.Generator(device="cuda").manual_seed(seed)
        image = get_pipe(request.lora_name, request.lora_scale)(prompt=request.prompt, negative_prompt=request.negative_prompt or None, width=request.width, height=request.height, num_inference_steps=request.steps, true_cfg_scale=request.guidance_scale, generator=generator).images[0]
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return {"seed": seed, "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"), "mime_type": "image/png"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@spaces.GPU(duration=120)
def generate_ui(prompt, lora_name, negative_prompt, width, height, steps, guidance, scale, seed):
    result = generate(GenerateRequest(prompt=prompt, lora_name=lora_name or None, negative_prompt=negative_prompt, width=int(width), height=int(height), steps=int(steps), guidance_scale=float(guidance), lora_scale=float(scale), seed=int(seed) if seed else None))
    from PIL import Image
    return Image.open(BytesIO(base64.b64decode(result["image_base64"]))), result["seed"]


with gr.Blocks(title="Qwen Image LoRA Studio") as demo:
    gr.Markdown("# Qwen Image LoRA Studio\nPrivate Qwen-Image-2512 LoRA training via ostris AI Toolkit.")
    with gr.Tab("Train"):
        files = gr.File(label="Training images", file_count="multiple", file_types=["image"], type="filepath")
        with gr.Row():
            name = gr.Textbox(label="Private LoRA name", placeholder="mysubject-v1")
            trigger = gr.Textbox(label="Trigger word", placeholder="mysubject")
        caption = gr.Textbox(label="Default caption", placeholder="photo of mysubject")
        steps = gr.Slider(500, 4000, value=1500, step=100, label="Training steps")
        train = gr.Button("Train and publish private LoRA", variant="primary")
        train_result = gr.Markdown()
        train.click(start_training, [files, name, trigger, caption, steps], train_result)
        check_name = gr.Textbox(label="Job name")
        gr.Button("Check status").click(status, check_name, train_result)
    with gr.Tab("Generate"):
        prompt = gr.Textbox(label="Prompt")
        lora = gr.Textbox(label="LoRA name (private model slug or owner/repo)")
        negative = gr.Textbox(label="Negative prompt")
        with gr.Row():
            width = gr.Slider(256, 2048, value=1024, step=32, label="Width")
            height = gr.Slider(256, 2048, value=1024, step=32, label="Height")
            infer_steps = gr.Slider(1, 80, value=28, step=1, label="Steps")
        with gr.Row():
            guidance = gr.Slider(0, 20, value=4, step=0.1, label="Guidance")
            scale = gr.Slider(0, 2, value=0.8, step=0.05, label="LoRA scale")
            seed = gr.Number(label="Seed (empty = random)", precision=0)
        run = gr.Button("Generate", variant="primary")
        image = gr.Image(label="Result")
        used_seed = gr.Number(label="Used seed", precision=0)
        run.click(generate_ui, [prompt, lora, negative, width, height, infer_steps, guidance, scale, seed], [image, used_seed])

demo.app.add_api_route("/v1/generate", generate, methods=["POST"], response_model=None)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0")
