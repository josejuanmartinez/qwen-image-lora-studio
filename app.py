import base64
import copy
import os
import re
import subprocess
import sys
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

# ZeroGPU must be imported before any package that may initialize CUDA.
try:
    import spaces
except ImportError:  # Allows local development outside Hugging Face Spaces.
    class _SpacesFallback:
        @staticmethod
        def GPU(*_args, **_kwargs):
            return lambda function: function
    spaces = _SpacesFallback()

import gradio as gr
import torch
import yaml
from diffusers import DiffusionPipeline
from fastapi import HTTPException
from huggingface_hub import HfApi
from pydantic import BaseModel, Field

BASE_MODEL = "Qwen/Qwen-Image-2512"
OWNER = os.getenv("HF_SPACE_OWNER", "jjmcarrascosa")
ROOT = Path(os.getenv("SPACE_WORKDIR", "/tmp/qwen-image-lora-studio"))
TOOLKIT = ROOT / "ai-toolkit"
JOBS = ROOT / "jobs"
JOBS.mkdir(parents=True, exist_ok=True)
TOOLKIT_INSTALL_MARKER = TOOLKIT / ".studio-dependencies-installed"
GALLERY_PAGE_SIZE = 12
TRAINING_PARAM_NAMES = (
    "steps", "rank", "alpha", "network_dropout", "learning_rate", "optimizer",
    "lr_scheduler", "batch_size", "gradient_accumulation", "max_grad_norm",
    "timestep_type", "train_dtype", "gradient_checkpointing", "cache_text_embeddings",
    "caption_dropout_rate", "token_dropout_rate", "shuffle_tokens", "keep_tokens",
    "num_repeats", "random_crop", "flip_x", "resolutions", "save_every",
    "max_saves", "save_dtype", "disable_sampling", "sample_every", "sample_prompt",
    "sample_negative", "sample_width", "sample_height", "sample_steps",
    "sample_guidance", "sample_seed", "walk_seed", "quantize", "qtype",
    "quantize_text_encoder", "qtype_text_encoder", "low_vram", "advanced_yaml",
)

pipe = None
loaded_lora = None
pipe_lock = threading.Lock()
toolkit_lock = threading.Lock()
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


def run_checked(command: list[str], cwd: Optional[Path] = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode:
        output = result.stdout[-12000:]
        raise RuntimeError(f"Command failed ({' '.join(command)}):\n{output}")
    return result.stdout


def ensure_toolkit() -> None:
    with toolkit_lock:
        if not (TOOLKIT / "run.py").exists():
            if TOOLKIT.exists():
                raise RuntimeError(f"AI Toolkit checkout is incomplete at {TOOLKIT}. Restart the Space to rebuild /tmp.")
            run_checked(["git", "clone", "--depth", "1", "https://github.com/ostris/ai-toolkit.git", str(TOOLKIT)])
        if not TOOLKIT_INSTALL_MARKER.exists():
            run_checked([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=TOOLKIT)
            TOOLKIT_INSTALL_MARKER.touch()


def deep_merge(base: dict, overrides: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def parse_overrides(value: str) -> dict:
    if not value or not value.strip():
        return {}
    parsed = yaml.safe_load(value)
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError("Advanced YAML must be a mapping of AI Toolkit sd_trainer settings.")
    return parsed


def parse_resolutions(value: str) -> list[int]:
    try:
        resolutions = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("Resolutions must be comma-separated integers, for example 512, 768, 1024.") from exc
    if not resolutions or any(item < 256 or item > 2048 for item in resolutions):
        raise ValueError("Choose one or more resolutions between 256 and 2048.")
    return resolutions


def qwen_config(run_name: str, image_dir: Path, trigger_word: str, options: dict) -> dict:
    steps = int(options["steps"])
    save_every = int(options["save_every"])
    sample_every = int(options["sample_every"])
    if steps < 1 or save_every < 1 or sample_every < 1:
        raise ValueError("Training steps, save interval, and sample interval must be positive integers.")
    if int(options["rank"]) < 1 or float(options["alpha"]) <= 0:
        raise ValueError("LoRA rank and alpha must be greater than zero.")
    if float(options["learning_rate"]) <= 0:
        raise ValueError("Learning rate must be greater than zero.")
    network = {
        "type": "lora",
        "linear": int(options["rank"]),
        "linear_alpha": float(options["alpha"]),
    }
    if options.get("network_dropout") is not None:
        network["dropout"] = float(options["network_dropout"])

    process = {
        "type": "sd_trainer",
        "training_folder": str(JOBS),
        "device": "cuda:0",
        "trigger_word": trigger_word,
        "network": network,
        "save": {
            "dtype": options["save_dtype"],
            "save_every": save_every,
            "max_step_saves_to_keep": int(options["max_saves"]),
        },
        "datasets": [{
            "folder_path": str(image_dir),
            "caption_ext": "txt",
            "caption_dropout_rate": float(options["caption_dropout_rate"]),
            "token_dropout_rate": float(options["token_dropout_rate"]),
            "shuffle_tokens": bool(options["shuffle_tokens"]),
            "keep_tokens": int(options["keep_tokens"]),
            "num_repeats": int(options["num_repeats"]),
            "random_crop": bool(options["random_crop"]),
            "flip_x": bool(options["flip_x"]),
            "cache_latents_to_disk": True,
            "resolution": parse_resolutions(options["resolutions"]),
        }],
        "train": {
            "batch_size": int(options["batch_size"]),
            "cache_text_embeddings": bool(options["cache_text_embeddings"]),
            "steps": steps,
            "gradient_accumulation": int(options["gradient_accumulation"]),
            "train_unet": True,
            "train_text_encoder": False,
            "gradient_checkpointing": bool(options["gradient_checkpointing"]),
            "noise_scheduler": "flowmatch",
            "optimizer": options["optimizer"],
            "lr_scheduler": options["lr_scheduler"],
            "lr": float(options["learning_rate"]),
            "max_grad_norm": float(options["max_grad_norm"]),
            "timestep_type": options["timestep_type"],
            "disable_sampling": bool(options["disable_sampling"]),
            "dtype": options["train_dtype"],
        },
        "model": {
            "name_or_path": BASE_MODEL,
            "arch": "qwen_image",
            "quantize": bool(options["quantize"]),
            "qtype": options["qtype"],
            "quantize_te": bool(options["quantize_text_encoder"]),
            "qtype_te": options["qtype_text_encoder"],
            "low_vram": bool(options["low_vram"]),
        },
        "sample": {
            "sampler": "flowmatch",
            "sample_every": sample_every,
            "sample_start_step": 0,
            "width": int(options["sample_width"]),
            "height": int(options["sample_height"]),
            "prompts": [options["sample_prompt"].strip() or f"{trigger_word}, portrait photo, detailed"],
            "neg": options["sample_negative"],
            "seed": int(options["sample_seed"]),
            "walk_seed": bool(options["walk_seed"]),
            "guidance_scale": float(options["sample_guidance"]),
            "sample_steps": int(options["sample_steps"]),
        },
    }
    process = deep_merge(process, parse_overrides(options.get("advanced_yaml", "")))

    # These values are owned by the Studio and cannot be redirected by an override.
    process["type"] = "sd_trainer"
    process["training_folder"] = str(JOBS)
    process["device"] = "cuda:0"
    process["trigger_word"] = trigger_word
    if not isinstance(process.get("datasets"), list) or not process["datasets"]:
        raise ValueError("Advanced YAML must leave at least one dataset configured.")
    if not isinstance(process["datasets"][0], dict):
        raise ValueError("The first dataset in Advanced YAML must be a mapping.")
    process["datasets"][0]["folder_path"] = str(image_dir)
    process["datasets"][0]["caption_ext"] = "txt"

    return {
        "job": "extension",
        "config": {
            "name": run_name,
            "process": [process],
        },
        "meta": {"name": run_name, "base_model": BASE_MODEL, "trigger_word": trigger_word, "private": True},
    }


def run_training(run_name: str, image_dir: Path, config: dict) -> None:
    job = jobs[run_name]
    try:
        hf_token()
        job["status"] = "installing AI Toolkit"
        ensure_toolkit()
        config_file = JOBS / f"{run_name}.yaml"
        config_file.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
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


def start_training(files, lora_name: str, trigger_word: str, caption: str, *param_values):
    if not files:
        raise gr.Error("Upload one or more training images.")
    if not trigger_word.strip():
        raise gr.Error("Provide a trigger word, such as mysubject.")
    try:
        run_name = slug(lora_name)
        options = dict(zip(TRAINING_PARAM_NAMES, param_values, strict=True))
        config = qwen_config(run_name, JOBS / run_name / "images", trigger_word.strip(), options)
    except (TypeError, ValueError, yaml.YAMLError) as exc:
        raise gr.Error(str(exc)) from exc
    image_dir = JOBS / run_name / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image, ImageOps
    image_caption = caption.strip() or trigger_word.strip()
    if trigger_word.strip().lower() not in image_caption.lower():
        image_caption = f"{trigger_word.strip()}, {image_caption}"
    for index, path in enumerate(files):
        source = Path(path[0] if isinstance(path, (tuple, list)) else path)
        target = image_dir / f"{index:04d}.png"
        try:
            with Image.open(source) as uploaded:
                ImageOps.exif_transpose(uploaded).convert("RGB").save(target, format="PNG")
        except Exception as exc:
            raise gr.Error(f"Could not read {source.name}: {exc}") from exc
        target.with_suffix(".txt").write_text(image_caption, encoding="utf-8")
    jobs[run_name] = {"status": "queued", "repo_id": lora_repo(run_name), "log": "Queued"}
    threading.Thread(target=run_training, args=(run_name, image_dir, config), daemon=True).start()
    return f"Queued `{run_name}`. It will publish privately to `{lora_repo(run_name)}` when complete."


def gallery_page(files, page: int = 1):
    files = list(files or [])
    page_count = max(1, (len(files) + GALLERY_PAGE_SIZE - 1) // GALLERY_PAGE_SIZE)
    page = min(max(1, int(page or 1)), page_count)
    start = (page - 1) * GALLERY_PAGE_SIZE
    items = [(path, Path(path).name) for path in files[start:start + GALLERY_PAGE_SIZE]]
    summary = f"Page {page} of {page_count} · {len(files)} image{'s' if len(files) != 1 else ''}"
    return items, page, summary, gr.update(interactive=page > 1), gr.update(interactive=page < page_count)


def add_gallery_files(new_files, existing_files):
    accumulated = list(existing_files or [])
    for path in new_files or []:
        path = str(path)
        if path not in accumulated:
            accumulated.append(path)
    items, page, summary, previous, following = gallery_page(accumulated, 1)
    return accumulated, items, page, summary, previous, following


def change_gallery_page(files, page: int, delta: int):
    return gallery_page(files, int(page or 1) + delta)


def clear_gallery():
    items, page, summary, previous, following = gallery_page([], 1)
    return [], items, page, summary, previous, following


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
    remove_background: bool = True


def generate(request: GenerateRequest):
    try:
        seed = int(request.seed if request.seed is not None else torch.seed() % (2**31 - 1))
        generator = torch.Generator(device="cuda").manual_seed(seed)
        image = get_pipe(request.lora_name, request.lora_scale)(prompt=request.prompt, negative_prompt=request.negative_prompt or None, width=request.width, height=request.height, num_inference_steps=request.steps, true_cfg_scale=request.guidance_scale, generator=generator).images[0]
        if request.remove_background:
            # Qwen Image produces RGB art; use segmentation to make a genuine alpha PNG.
            # The rembg model downloads once on first request and remains cached by the Space.
            from rembg import remove
            image = remove(image).convert("RGBA")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return {"seed": seed, "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"), "mime_type": "image/png"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@spaces.GPU(duration=120)
def generate_ui(prompt, lora_name, negative_prompt, width, height, steps, guidance, scale, seed):
    result = generate(GenerateRequest(prompt=prompt, lora_name=lora_name or None, negative_prompt=negative_prompt, width=int(width), height=int(height), steps=int(steps), guidance_scale=float(guidance), lora_scale=float(scale), seed=int(seed) if seed else None, remove_background=True))
    from PIL import Image
    return Image.open(BytesIO(base64.b64decode(result["image_base64"]))), result["seed"]


with gr.Blocks(title="Qwen Image LoRA Studio") as demo:
    gr.Markdown("# Qwen Image LoRA Studio\nPrivate Qwen-Image-2512 LoRA training via ostris AI Toolkit.")
    with gr.Tab("Train"):
        image_files = gr.State([])
        gallery_page_number = gr.State(1)
        with gr.Row():
            upload = gr.UploadButton(
                "Add training images",
                file_count="multiple",
                file_types=["image"],
                type="filepath",
                variant="primary",
            )
            clear_images = gr.Button("Clear images")
        training_gallery = gr.Gallery(
            label="Training images",
            value=[],
            columns=4,
            rows=3,
            height=600,
            object_fit="cover",
            preview=True,
            interactive=False,
        )
        with gr.Row():
            previous_page = gr.Button("Previous", interactive=False)
            gallery_summary = gr.Markdown("Page 1 of 1 · 0 images")
            next_page = gr.Button("Next", interactive=False)
        gallery_outputs = [
            training_gallery,
            gallery_page_number,
            gallery_summary,
            previous_page,
            next_page,
        ]
        upload.upload(
            add_gallery_files,
            [upload, image_files],
            [image_files, *gallery_outputs],
        )
        previous_page.click(
            lambda current_files, current_page: change_gallery_page(current_files, current_page, -1),
            [image_files, gallery_page_number],
            gallery_outputs,
        )
        next_page.click(
            lambda current_files, current_page: change_gallery_page(current_files, current_page, 1),
            [image_files, gallery_page_number],
            gallery_outputs,
        )
        clear_images.click(clear_gallery, outputs=[image_files, *gallery_outputs])

        with gr.Row():
            name = gr.Textbox(label="Private LoRA name", placeholder="mysubject-v1")
            trigger = gr.Textbox(label="Trigger word", placeholder="mysubject")
        caption = gr.Textbox(label="Default caption", placeholder="photo of mysubject")
        steps = gr.Slider(500, 4000, value=1500, step=100, label="Training steps")

        with gr.Accordion("LoRA network", open=False):
            gr.Markdown("Defaults follow the Qwen Image 24 GB AI Toolkit recipe.")
            with gr.Row():
                rank = gr.Slider(1, 256, value=16, step=1, label="Rank / linear dimension")
                alpha = gr.Slider(0.1, 256, value=16, step=0.1, label="Linear alpha")
                network_dropout = gr.Slider(0, 1, value=0, step=0.01, label="Network dropout")

        with gr.Accordion("Optimizer and training", open=False):
            with gr.Row():
                learning_rate = gr.Number(value=0.0001, label="Learning rate")
                optimizer = gr.Dropdown(
                    ["adamw8bit", "adamw", "adafactor", "prodigy", "prodigy8bit"],
                    value="adamw8bit",
                    allow_custom_value=True,
                    label="Optimizer",
                )
                lr_scheduler = gr.Dropdown(
                    ["constant", "linear", "cosine", "cosine_with_restarts", "polynomial"],
                    value="constant",
                    allow_custom_value=True,
                    label="LR scheduler",
                )
            with gr.Row():
                batch_size = gr.Slider(1, 16, value=1, step=1, label="Batch size")
                gradient_accumulation = gr.Slider(1, 32, value=1, step=1, label="Gradient accumulation")
                max_grad_norm = gr.Number(value=1.0, label="Max gradient norm")
            with gr.Row():
                timestep_type = gr.Dropdown(
                    ["sigmoid", "linear", "lognorm_blend", "next_sample", "weighted", "one_step"],
                    value="sigmoid",
                    allow_custom_value=True,
                    label="Timestep type",
                )
                train_dtype = gr.Dropdown(["bf16", "fp16", "fp32"], value="bf16", label="Training dtype")
            with gr.Row():
                gradient_checkpointing = gr.Checkbox(True, label="Gradient checkpointing")
                cache_text_embeddings = gr.Checkbox(True, label="Cache text embeddings")

        with gr.Accordion("Dataset", open=False):
            with gr.Row():
                caption_dropout_rate = gr.Slider(0, 1, value=0.05, step=0.01, label="Caption dropout")
                token_dropout_rate = gr.Slider(0, 1, value=0, step=0.01, label="Token dropout")
                shuffle_tokens = gr.Checkbox(False, label="Shuffle caption tokens")
                keep_tokens = gr.Slider(0, 32, value=0, step=1, label="Keep leading tokens")
            with gr.Row():
                num_repeats = gr.Slider(1, 100, value=1, step=1, label="Dataset repeats")
                random_crop = gr.Checkbox(False, label="Random crop")
                flip_x = gr.Checkbox(False, label="Horizontal flip")
                resolutions = gr.Textbox(value="512, 768, 1024", label="Resolution buckets")

        with gr.Accordion("Saving and samples", open=False):
            with gr.Row():
                save_every = gr.Number(value=250, precision=0, label="Save every N steps")
                max_saves = gr.Slider(1, 20, value=2, step=1, label="Checkpoints to keep")
                save_dtype = gr.Dropdown(["float16", "bf16", "float32"], value="float16", label="Saved weight dtype")
                disable_sampling = gr.Checkbox(False, label="Disable samples")
            with gr.Row():
                sample_every = gr.Number(value=250, precision=0, label="Sample every N steps")
                sample_prompt = gr.Textbox(label="Sample prompt", placeholder="[trigger], portrait photo, detailed")
                sample_negative = gr.Textbox(value="blurry, low quality", label="Sample negative prompt")
            with gr.Row():
                sample_width = gr.Slider(256, 2048, value=1024, step=32, label="Sample width")
                sample_height = gr.Slider(256, 2048, value=1024, step=32, label="Sample height")
                sample_steps = gr.Slider(1, 80, value=28, step=1, label="Sample steps")
                sample_guidance = gr.Slider(0, 20, value=4, step=0.1, label="Sample guidance")
            with gr.Row():
                sample_seed = gr.Number(value=42, precision=0, label="Sample seed")
                walk_seed = gr.Checkbox(False, label="Walk seed between prompts")

        with gr.Accordion("Model and memory", open=False):
            with gr.Row():
                quantize = gr.Checkbox(True, label="Quantize transformer")
                qtype = gr.Textbox(
                    value="uint4|ostris/accuracy_recovery_adapters/qwen_image_2512_torchao_uint4.safetensors",
                    label="Transformer quantization / ARA",
                )
            with gr.Row():
                quantize_text_encoder = gr.Checkbox(True, label="Quantize text encoder")
                qtype_text_encoder = gr.Textbox(value="qfloat8", label="Text encoder quantization")
                low_vram = gr.Checkbox(True, label="Low VRAM mode")

        with gr.Accordion("All AI Toolkit parameters (advanced)", open=False):
            gr.Markdown(
                "Optional YAML is deep-merged into the `sd_trainer` process. This exposes any "
                "AI Toolkit parameter not listed above. Managed paths, device, process type, and trigger word stay protected."
            )
            advanced_yaml = gr.Code(
                language="yaml",
                lines=12,
                label="Optional sd_trainer YAML overrides",
                value="",
            )

        train = gr.Button("Train and publish private LoRA", variant="primary")
        train_result = gr.Markdown()
        training_controls = [
            steps, rank, alpha, network_dropout, learning_rate, optimizer, lr_scheduler,
            batch_size, gradient_accumulation, max_grad_norm, timestep_type, train_dtype,
            gradient_checkpointing, cache_text_embeddings, caption_dropout_rate,
            token_dropout_rate, shuffle_tokens, keep_tokens, num_repeats, random_crop,
            flip_x, resolutions, save_every, max_saves, save_dtype, disable_sampling,
            sample_every, sample_prompt, sample_negative, sample_width, sample_height,
            sample_steps, sample_guidance, sample_seed, walk_seed, quantize, qtype,
            quantize_text_encoder, qtype_text_encoder, low_vram, advanced_yaml,
        ]
        train.click(start_training, [image_files, name, trigger, caption, *training_controls], train_result)
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
