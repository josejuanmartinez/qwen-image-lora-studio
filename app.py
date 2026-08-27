import base64
import copy
import json
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
SWIN2SR_MODEL = "caidas/swin2SR-lightweight-x2-64"
OWNER = os.getenv("HF_SPACE_OWNER", "jjmcarrascosa")
ROOT = Path(os.getenv("SPACE_WORKDIR", "/tmp/qwen-image-lora-studio"))
TOOLKIT = ROOT / "ai-toolkit"
TOOLKIT_VENV = ROOT / ".ai-toolkit-venv"
JOBS = ROOT / "jobs"
JOBS.mkdir(parents=True, exist_ok=True)
LATEST_JOB_FILE = JOBS / "latest-job.txt"
TOOLKIT_INSTALL_MARKER = TOOLKIT_VENV / ".studio-dependencies-installed-v4"
TOOLKIT_COMPAT_PACKAGES = ["kernels==0.12.3"]
GALLERY_PAGE_SIZE = 12
MAX_JOB_LOG_CHARS = 100_000
ACTIVE_JOB_STATUSES = {"preparing images", "queued", "installing AI Toolkit", "training", "publishing"}
APP_CSS = """
.training-log-console textarea {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
    overflow-y: auto !important;
    resize: none !important;
}
.transparent-result .image-container,
.transparent-result .wrap {
    background-color: #f4f4f4 !important;
    background-image:
        linear-gradient(45deg, #d8d8d8 25%, transparent 25%),
        linear-gradient(-45deg, #d8d8d8 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #d8d8d8 75%),
        linear-gradient(-45deg, transparent 75%, #d8d8d8 75%) !important;
    background-position: 0 0, 0 10px, 10px -10px, -10px 0 !important;
    background-size: 20px 20px !important;
}
"""
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
swin2sr_processor = None
swin2sr_model = None
pipe_lock = threading.Lock()
swin2sr_lock = threading.Lock()
toolkit_lock = threading.Lock()
jobs_lock = threading.RLock()
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


def job_log_path(run_name: str) -> Path:
    return JOBS / slug(run_name) / "training.log"


def job_metadata_path(run_name: str) -> Path:
    return JOBS / slug(run_name) / "job.json"


def persist_job_metadata(run_name: str) -> None:
    job = jobs.get(run_name)
    if job is None:
        return
    path = job_metadata_path(run_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in job.items() if key != "log"}
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_job_metadata(run_name: str) -> Optional[dict]:
    run_name = slug(run_name)
    with jobs_lock:
        if run_name in jobs:
            return jobs[run_name]
        path = job_metadata_path(run_name)
        if not path.exists():
            return None
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        if not isinstance(job, dict):
            return None
        jobs[run_name] = job
        return job


def latest_job_name() -> str:
    try:
        candidate = slug(LATEST_JOB_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    return candidate if job_metadata_path(candidate).exists() else ""


def read_job_log(run_name: str) -> str:
    try:
        content = job_log_path(run_name).read_text(encoding="utf-8", errors="replace")
    except OSError:
        with jobs_lock:
            content = jobs.get(run_name, {}).get("log", "")
    return content[-MAX_JOB_LOG_CHARS:] or "No log output yet."


def append_job_log(run_name: str, message: str) -> None:
    with jobs_lock:
        job = jobs.get(run_name)
        if job is None:
            return
        job["log"] = (job.get("log", "") + message)[-MAX_JOB_LOG_CHARS:]
        job["updated_at"] = time.time()
        path = job_log_path(run_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(message)
        if path.stat().st_size > MAX_JOB_LOG_CHARS * 2:
            path.write_text(read_job_log(run_name), encoding="utf-8")


def set_job_status(run_name: str, status_value: str, message: Optional[str] = None) -> None:
    with jobs_lock:
        jobs[run_name]["status"] = status_value
        jobs[run_name]["updated_at"] = time.time()
        persist_job_metadata(run_name)
    if message:
        append_job_log(run_name, f"\n[{time.strftime('%H:%M:%S')}] {message}\n")


def run_checked(command: list[str], cwd: Optional[Path] = None, on_output=None) -> str:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    output_tail = ""
    assert process.stdout is not None
    for line in process.stdout:
        output_tail = (output_tail + line)[-12000:]
        if on_output:
            on_output(line)
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"Command failed ({' '.join(command)}):\n{output_tail}")
    return output_tail


def toolkit_python() -> Path:
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    executable = "python.exe" if os.name == "nt" else "python"
    return TOOLKIT_VENV / scripts_dir / executable


def toolkit_torch_version() -> str:
    output = run_checked(
        [
            str(toolkit_python()),
            "-c",
            "import torch; print(torch.__version__.split('+', 1)[0])",
        ],
        cwd=TOOLKIT,
    )
    version = output.strip().splitlines()[-1]
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:[a-z]+\d+)?", version):
        raise RuntimeError(f"Could not determine a compatible torchaudio version from PyTorch {version!r}.")
    return version


def compatible_torchaudio_version(torch_version: str) -> str:
    major, minor, _patch = (int(part) for part in torch_version.split(".")[:3])
    # TorchAudio 2.11 adopted PyTorch's stable ABI and supports PyTorch 2.11+
    # without requiring a same-numbered TorchAudio release.
    if (major, minor) >= (2, 11):
        return "2.11.0"
    if torch_version == "2.0.1":
        return "2.0.2"
    if torch_version == "2.0.0":
        return "2.0.1"
    return torch_version


def ensure_toolkit(run_name: Optional[str] = None) -> None:
    emit = (lambda line: append_job_log(run_name, line)) if run_name else None
    with toolkit_lock:
        if not (TOOLKIT / "run.py").exists():
            if TOOLKIT.exists():
                raise RuntimeError(f"AI Toolkit checkout is incomplete at {TOOLKIT}. Restart the Space to rebuild /tmp.")
            run_checked(
                ["git", "clone", "--depth", "1", "https://github.com/ostris/ai-toolkit.git", str(TOOLKIT)],
                on_output=emit,
            )
        if not toolkit_python().exists():
            run_checked(
                [sys.executable, "-m", "venv", "--system-site-packages", str(TOOLKIT_VENV)],
                on_output=emit,
            )
        if not TOOLKIT_INSTALL_MARKER.exists():
            run_checked(
                [str(toolkit_python()), "-m", "pip", "install", "-r", "requirements.txt"],
                cwd=TOOLKIT,
                on_output=emit,
            )
            # Transformers 5.5.x expects the kernels 0.12 LayerRepository API. The
            # Space inference environment may provide a newer, incompatible release
            # through --system-site-packages, so shadow it inside the toolkit venv.
            run_checked(
                [str(toolkit_python()), "-m", "pip", "install", *TOOLKIT_COMPAT_PACKAGES],
                cwd=TOOLKIT,
                on_output=emit,
            )
            torch_version = toolkit_torch_version()
            torchaudio_version = compatible_torchaudio_version(torch_version)
            if emit:
                emit(f"Installing torchaudio {torchaudio_version} for PyTorch {torch_version}\n")
            run_checked(
                [
                    str(toolkit_python()),
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    f"torchaudio=={torchaudio_version}",
                ],
                cwd=TOOLKIT,
                on_output=emit,
            )
            run_checked(
                [
                    str(toolkit_python()),
                    "-c",
                    "import torchaudio; "
                    "from transformers import T5Tokenizer, T5EncoderModel, UMT5EncoderModel; "
                    "print('AI Toolkit dependency check passed')",
                ],
                cwd=TOOLKIT,
                on_output=emit,
            )
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
            "prompts": [(options.get("sample_prompt") or "").strip() or f"{trigger_word}, portrait photo, detailed"],
            "neg": options.get("sample_negative") or "",
            "seed": int(options.get("sample_seed") if options.get("sample_seed") is not None else 42),
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
    try:
        hf_token()
        set_job_status(run_name, "installing AI Toolkit", "Preparing the isolated AI Toolkit environment…")
        ensure_toolkit(run_name)
        config_file = JOBS / f"{run_name}.yaml"
        config_file.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        set_job_status(run_name, "training", f"Starting AI Toolkit with {config_file.name}…")
        run_checked(
            [str(toolkit_python()), "run.py", str(config_file)],
            cwd=TOOLKIT,
            on_output=lambda line: append_job_log(run_name, line),
        )
        weights = sorted((JOBS / run_name).rglob("*.safetensors"), key=lambda p: p.stat().st_mtime)
        if not weights:
            raise RuntimeError("AI Toolkit completed but did not produce a .safetensors LoRA.")
        set_job_status(run_name, "publishing", "Training completed. Publishing the adapter privately…")
        api = HfApi(token=hf_token())
        repo_id = lora_repo(run_name)
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=str(weights[-1]), path_in_repo="adapter.safetensors", repo_id=repo_id, repo_type="model", commit_message=f"Publish private Qwen Image LoRA {run_name}")
        api.upload_file(path_or_fileobj=str(config_file), path_in_repo="training_config.yaml", repo_id=repo_id, repo_type="model", commit_message="Add training configuration")
        with jobs_lock:
            jobs[run_name]["repo_id"] = repo_id
            persist_job_metadata(run_name)
        set_job_status(run_name, "published", f"Published privately to {repo_id}")
    except Exception as exc:
        set_job_status(run_name, "failed", f"ERROR: {str(exc)[-12000:]}")


def start_training(files, lora_name: str, trigger_word: str, caption: str, *param_values):
    lora_name = (lora_name or "").strip()
    trigger_word = (trigger_word or "").strip()
    caption = caption or ""
    if not files:
        raise gr.Error("Upload one or more training images.")
    if not lora_name:
        raise gr.Error("Enter a required job / private LoRA name before training.")
    if not trigger_word:
        raise gr.Error("Provide a trigger word, such as mysubject.")
    try:
        run_name = slug(lora_name)
        options = dict(zip(TRAINING_PARAM_NAMES, param_values, strict=True))
        image_dir = JOBS / run_name / f"images-{time.time_ns()}"
        config = qwen_config(run_name, image_dir, trigger_word, options)
    except (TypeError, ValueError, yaml.YAMLError) as exc:
        raise gr.Error(str(exc)) from exc
    with jobs_lock:
        active_jobs = [name for name, job in jobs.items() if job.get("status") in ACTIVE_JOB_STATUSES]
        if active_jobs:
            raise gr.Error(f"Training is already running for `{active_jobs[0]}`. Wait for it to finish before starting another job.")
        jobs[run_name] = {
            "status": "preparing images",
            "repo_id": lora_repo(run_name),
            "log": f"[{time.strftime('%H:%M:%S')}] Preparing uploaded images…\n",
            "updated_at": time.time(),
        }
        job_log_path(run_name).parent.mkdir(parents=True, exist_ok=True)
        job_log_path(run_name).write_text(jobs[run_name]["log"], encoding="utf-8")
        persist_job_metadata(run_name)
        LATEST_JOB_FILE.write_text(run_name, encoding="utf-8")
    image_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image, ImageOps
    image_caption = caption.strip() or trigger_word
    if trigger_word.lower() not in image_caption.lower():
        image_caption = f"{trigger_word}, {image_caption}"
    for index, path in enumerate(files):
        source = Path(path[0] if isinstance(path, (tuple, list)) else path)
        target = image_dir / f"{index:04d}.png"
        try:
            with Image.open(source) as uploaded:
                ImageOps.exif_transpose(uploaded).convert("RGB").save(target, format="PNG")
        except Exception as exc:
            set_job_status(run_name, "failed", f"ERROR: Could not read {source.name}: {exc}")
            raise gr.Error(f"Could not read {source.name}: {exc}") from exc
        target.with_suffix(".txt").write_text(image_caption, encoding="utf-8")
    set_job_status(run_name, "queued", f"Queued {len(files)} training images.")
    threading.Thread(target=run_training, args=(run_name, image_dir, config), daemon=True).start()
    return (
        f"### ⏳ Queued `{run_name}`\nIt will publish privately to `{lora_repo(run_name)}` when complete.",
        run_name,
        read_job_log(run_name),
        gr.update(interactive=False, value="Training in progress…"),
    )


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


def training_button_state(lora_name: str, trigger_word: str, files):
    with jobs_lock:
        any_active = any(job.get("status") in ACTIVE_JOB_STATUSES for job in jobs.values())
    if any_active:
        return gr.update(interactive=False, value="Training in progress…")
    if not (lora_name or "").strip():
        return gr.update(interactive=False, value="Enter a job name to train")
    if not (trigger_word or "").strip():
        return gr.update(interactive=False, value="Enter a trigger word to train")
    if not files:
        return gr.update(interactive=False, value="Add training images to continue")
    return gr.update(interactive=True, value="Train and publish private LoRA")


def job_view(lora_name: str, form_name: str = "", trigger_word: str = "", files=None):
    try:
        run_name = slug(lora_name) if lora_name else latest_job_name()
        loaded = load_job_metadata(run_name) if run_name else None
        job = copy.deepcopy(loaded) if loaded else None
    except ValueError:
        run_name, job = "", None
    button = training_button_state(form_name, trigger_word, files)
    if job is None:
        return "### No local job found", "Enter a job name or start training to view its live log.", button, ""
    status_value = job.get("status", "unknown")
    icon = "✅" if status_value == "published" else "❌" if status_value == "failed" else "⏳"
    repo_id = job.get("repo_id", "")
    summary = f"### {icon} `{run_name}` — {status_value}\nPrivate destination: `{repo_id}`"
    return summary, read_job_log(run_name), button, run_name


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
    upscale_to_2k: bool = False


def target_2k_size(image):
    target_long_edge = 2048
    current_long_edge = max(image.size)
    if current_long_edge >= target_long_edge:
        return image.size
    scale = target_long_edge / current_long_edge
    return (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )


def lanczos_upscale_to_2k(image):
    from PIL import Image
    return image.resize(target_2k_size(image), Image.Resampling.LANCZOS)


def get_swin2sr():
    global swin2sr_processor, swin2sr_model
    with swin2sr_lock:
        if swin2sr_model is None:
            from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution
            processor = AutoImageProcessor.from_pretrained(SWIN2SR_MODEL)
            model = Swin2SRForImageSuperResolution.from_pretrained(SWIN2SR_MODEL)
            model.eval().to("cuda")
            swin2sr_processor = processor
            swin2sr_model = model
        return swin2sr_processor, swin2sr_model


def run_swin2sr_tiled(image, tile_size=512, overlap=32):
    """Run learned x2 super-resolution in contextual tiles to control VRAM use."""
    from PIL import Image

    processor, model = get_swin2sr()
    output = Image.new("RGB", (image.width * 2, image.height * 2))
    core_size = tile_size - (overlap * 2)
    for top in range(0, image.height, core_size):
        for left in range(0, image.width, core_size):
            core_right = min(left + core_size, image.width)
            core_bottom = min(top + core_size, image.height)
            tile_left = max(0, left - overlap)
            tile_top = max(0, top - overlap)
            tile_right = min(image.width, core_right + overlap)
            tile_bottom = min(image.height, core_bottom + overlap)
            tile = image.crop((tile_left, tile_top, tile_right, tile_bottom))

            inputs = processor(images=tile, return_tensors="pt")
            inputs = {name: value.to("cuda") for name, value in inputs.items()}
            with torch.inference_mode():
                reconstruction = model(**inputs).reconstruction
            pixels = (
                reconstruction[0]
                .detach()
                .float()
                .clamp_(0, 1)
                .mul_(255)
                .round_()
                .byte()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            enhanced_tile = Image.fromarray(pixels, mode="RGB")
            expected_tile_size = (tile.width * 2, tile.height * 2)
            if enhanced_tile.size != expected_tile_size:
                enhanced_tile = enhanced_tile.resize(expected_tile_size, Image.Resampling.LANCZOS)

            crop_box = (
                (left - tile_left) * 2,
                (top - tile_top) * 2,
                (core_right - tile_left) * 2,
                (core_bottom - tile_top) * 2,
            )
            output.paste(enhanced_tile.crop(crop_box), (left * 2, top * 2))
    return output


def upscale_image_to_2k(image):
    """Use Swin2SR for RGB detail and preserve any transparency independently."""
    from PIL import Image

    final_size = target_2k_size(image)
    if final_size == image.size:
        return image, "unchanged (already 2K)"

    # The checkpoint is a native x2 model. Normalize its input to half of the
    # desired dimensions, then make the final one-pixel adjustment if needed.
    model_input_size = (
        max(1, (final_size[0] + 1) // 2),
        max(1, (final_size[1] + 1) // 2),
    )
    rgb = image.convert("RGB")
    if rgb.size != model_input_size:
        rgb = rgb.resize(model_input_size, Image.Resampling.LANCZOS)
    enhanced = run_swin2sr_tiled(rgb)
    if enhanced.size != final_size:
        enhanced = enhanced.resize(final_size, Image.Resampling.LANCZOS)

    if "A" in image.getbands():
        alpha = image.getchannel("A").resize(final_size, Image.Resampling.LANCZOS)
        enhanced = enhanced.convert("RGBA")
        enhanced.putalpha(alpha)
    return enhanced, f"Swin2SR x2 ({SWIN2SR_MODEL})"


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
        upscaler = None
        upscale_warning = None
        if request.upscale_to_2k:
            try:
                image, upscaler = upscale_image_to_2k(image)
            except Exception as upscale_error:
                # Generation should remain usable if the optional model cannot
                # download, initialize, or fit in the currently available VRAM.
                upscale_warning = f"Swin2SR unavailable; used Lanczos fallback: {upscale_error}"
                print(upscale_warning, flush=True)
                image = lanczos_upscale_to_2k(image)
                upscaler = "Lanczos fallback"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        has_transparency = image.mode == "RGBA" and image.getextrema()[3][0] < 255
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return {
            "seed": seed,
            "width": image.width,
            "height": image.height,
            "transparent": has_transparency,
            "upscaler": upscaler,
            "upscale_warning": upscale_warning,
            "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "mime_type": "image/png",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@spaces.GPU(duration=120)
def generate_ui(prompt, lora_name, negative_prompt, width, height, steps, guidance, scale, seed, remove_background, upscale_to_2k):
    result = generate(GenerateRequest(
        prompt=prompt,
        lora_name=lora_name or None,
        negative_prompt=negative_prompt,
        width=int(width),
        height=int(height),
        steps=int(steps),
        guidance_scale=float(guidance),
        lora_scale=float(scale),
        seed=int(seed) if seed is not None else None,
        remove_background=bool(remove_background),
        upscale_to_2k=bool(upscale_to_2k),
    ))
    from PIL import Image
    transparency = "transparent PNG" if result["transparent"] else "opaque PNG"
    upscaler = f" · {result['upscaler']}" if result["upscaler"] else ""
    warning = f"\n\n⚠️ {result['upscale_warning']}" if result["upscale_warning"] else ""
    details = f"**{result['width']} × {result['height']}** · {transparency} · seed `{result['seed']}`{upscaler}{warning}"
    return Image.open(BytesIO(base64.b64decode(result["image_base64"]))), result["seed"], details


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
            name = gr.Textbox(
                label="Job / private LoRA name (required)",
                placeholder="mysubject-v1",
                info="Required before training. This also becomes the private model repository name.",
            )
            trigger = gr.Textbox(
                label="Trigger word",
                placeholder="mysubject",
                info="Unique word used in captions and prompts to activate this LoRA.",
            )
        with gr.Row():
            caption = gr.Textbox(
                label="Common style caption",
                placeholder="photo of mysubject",
                info="Shared description saved with every training image. The trigger word is added automatically if missing.",
            )
            sample_prompt = gr.Textbox(
                label="Verification image prompt",
                placeholder="mysubject, portrait photo, detailed",
                info="Prompt used only to create progress images during training; it does not label the dataset.",
            )
        steps = gr.Slider(
            500, 4000, value=1500, step=100, label="Training steps",
            info="Total optimizer updates. More steps can learn more detail but may overfit.",
        )

        with gr.Accordion("LoRA network", open=False):
            gr.Markdown("Defaults follow the Qwen Image 24 GB AI Toolkit recipe.")
            with gr.Row():
                rank = gr.Slider(
                    1, 256, value=16, step=1, label="Rank / linear dimension",
                    info="LoRA capacity. Higher ranks can capture more detail but use more VRAM and produce larger files.",
                )
                alpha = gr.Slider(
                    0.1, 256, value=16, step=0.1, label="Linear alpha",
                    info="Scales LoRA updates during training; matching alpha to rank is a common baseline.",
                )
                network_dropout = gr.Slider(
                    0, 1, value=0, step=0.01, label="Network dropout",
                    info="Randomly drops LoRA activations to reduce overfitting. Zero disables it.",
                )

        with gr.Accordion("Optimizer and training", open=False):
            with gr.Row():
                learning_rate = gr.Number(
                    value=0.0001, label="Learning rate",
                    info="Size of each parameter update. Lower values train more gently.",
                )
                optimizer = gr.Dropdown(
                    ["adamw8bit", "adamw", "adafactor", "prodigy", "prodigy8bit"],
                    value="adamw8bit",
                    allow_custom_value=True,
                    label="Optimizer",
                    info="Algorithm that updates LoRA weights; adamw8bit is the memory-efficient default.",
                )
                lr_scheduler = gr.Dropdown(
                    ["constant", "linear", "cosine", "cosine_with_restarts", "polynomial"],
                    value="constant",
                    allow_custom_value=True,
                    label="LR scheduler",
                    info="How the learning rate changes over the course of training.",
                )
            with gr.Row():
                batch_size = gr.Slider(
                    1, 16, value=1, step=1, label="Batch size",
                    info="Images processed simultaneously. Increase only when VRAM allows.",
                )
                gradient_accumulation = gr.Slider(
                    1, 32, value=1, step=1, label="Gradient accumulation",
                    info="Combines several batches before updating weights, simulating a larger batch.",
                )
                max_grad_norm = gr.Number(
                    value=1.0, label="Max gradient norm",
                    info="Clips unusually large gradients for training stability.",
                )
            with gr.Row():
                timestep_type = gr.Dropdown(
                    ["sigmoid", "linear", "lognorm_blend", "next_sample", "weighted", "one_step"],
                    value="sigmoid",
                    allow_custom_value=True,
                    label="Timestep type",
                    info="Controls how diffusion timesteps are sampled during training.",
                )
                train_dtype = gr.Dropdown(
                    ["bf16", "fp16", "fp32"], value="bf16", label="Training dtype",
                    info="Numeric precision used for training; bf16 is recommended on modern GPUs.",
                )
            with gr.Row():
                gradient_checkpointing = gr.Checkbox(
                    True, label="Gradient checkpointing",
                    info="Recomputes activations to substantially reduce VRAM usage.",
                )
                cache_text_embeddings = gr.Checkbox(
                    True, label="Cache text embeddings",
                    info="Encodes captions once to save VRAM and speed up repeated training steps.",
                )

        with gr.Accordion("Dataset", open=False):
            with gr.Row():
                caption_dropout_rate = gr.Slider(
                    0, 1, value=0.05, step=0.01, label="Caption dropout",
                    info="Chance of training an image without its caption to reduce prompt dependence.",
                )
                token_dropout_rate = gr.Slider(
                    0, 1, value=0, step=0.01, label="Token dropout",
                    info="Chance of dropping individual caption tokens during training.",
                )
                shuffle_tokens = gr.Checkbox(
                    False, label="Shuffle caption tokens",
                    info="Randomizes comma-separated caption tags while training.",
                )
                keep_tokens = gr.Slider(
                    0, 32, value=0, step=1, label="Keep leading tokens",
                    info="Number of leading caption tokens never shuffled or dropped.",
                )
            with gr.Row():
                num_repeats = gr.Slider(
                    1, 100, value=1, step=1, label="Dataset repeats",
                    info="Relative frequency with which this dataset is sampled.",
                )
                random_crop = gr.Checkbox(
                    False, label="Random crop",
                    info="Uses random image regions instead of a fixed centered crop.",
                )
                flip_x = gr.Checkbox(
                    False, label="Horizontal flip",
                    info="Randomly mirrors images; avoid when direction, text, or asymmetry matters.",
                )
                resolutions = gr.Textbox(
                    value="512, 768, 1024", label="Resolution buckets",
                    info="Comma-separated training sizes used to bucket different image aspect ratios.",
                )

        with gr.Accordion("Saving and samples", open=False):
            with gr.Row():
                save_every = gr.Number(
                    value=250, precision=0, label="Save every N steps",
                    info="Interval between intermediate LoRA checkpoints.",
                )
                max_saves = gr.Slider(
                    1, 20, value=2, step=1, label="Checkpoints to keep",
                    info="Maximum intermediate checkpoints retained on disk.",
                )
                save_dtype = gr.Dropdown(
                    ["float16", "bf16", "float32"], value="float16", label="Saved weight dtype",
                    info="Precision of the saved adapter; float16 is compact and broadly compatible.",
                )
                disable_sampling = gr.Checkbox(
                    False, label="Disable samples",
                    info="Skips verification images to reduce training interruptions.",
                )
            with gr.Row():
                sample_every = gr.Number(
                    value=250, precision=0, label="Verify every N steps",
                    info="Interval between generated verification images.",
                )
                sample_negative = gr.Textbox(
                    value="blurry, low quality", label="Verification negative prompt",
                    info="Qualities to discourage in verification images.",
                )
            with gr.Row():
                sample_width = gr.Slider(
                    256, 2048, value=1024, step=32, label="Verification width",
                    info="Width of progress images generated during training.",
                )
                sample_height = gr.Slider(
                    256, 2048, value=1024, step=32, label="Verification height",
                    info="Height of progress images generated during training.",
                )
                sample_steps = gr.Slider(
                    1, 80, value=28, step=1, label="Verification steps",
                    info="Denoising steps used for each verification image.",
                )
                sample_guidance = gr.Slider(
                    0, 20, value=4, step=0.1, label="Verification guidance",
                    info="How strongly verification images follow the prompt.",
                )
            with gr.Row():
                sample_seed = gr.Number(
                    value=42, precision=0, label="Verification seed",
                    info="Fixed random seed for comparable progress images.",
                )
                walk_seed = gr.Checkbox(
                    False, label="Walk seed between prompts",
                    info="Increments the seed for each verification prompt.",
                )

        with gr.Accordion("Model and memory", open=False):
            with gr.Row():
                quantize = gr.Checkbox(
                    True, label="Quantize transformer",
                    info="Loads lower-precision transformer weights to reduce VRAM usage.",
                )
                qtype = gr.Textbox(
                    value="uint4|ostris/accuracy_recovery_adapters/qwen_image_2512_torchao_uint4.safetensors",
                    label="Transformer quantization / ARA",
                    info="Quantization format and optional accuracy-recovery adapter path.",
                )
            with gr.Row():
                quantize_text_encoder = gr.Checkbox(
                    True, label="Quantize text encoder",
                    info="Uses lower-precision text-encoder weights to save VRAM.",
                )
                qtype_text_encoder = gr.Textbox(
                    value="qfloat8", label="Text encoder quantization",
                    info="Quantization format used for the text encoder.",
                )
                low_vram = gr.Checkbox(
                    True, label="Low VRAM mode",
                    info="Enables AI Toolkit memory-saving behavior for limited GPUs.",
                )

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

        with gr.Accordion("Look up an existing job", open=False):
            with gr.Row():
                check_name = gr.Textbox(
                    label="Existing job name",
                    info="Local job name whose current status and log you want to display.",
                )
                check_status = gr.Button("Check status")

        active_job = gr.State("")
        train = gr.Button("Enter a job name to train", variant="primary", interactive=False)
        training_status = gr.Markdown("### Ready to train")
        training_log = gr.Textbox(
            value="Training output will appear here live.",
            lines=50,
            max_lines=50,
            label="Live training log",
            info="Fixed-height console. It follows new output automatically; scroll up to pause auto-scrolling and inspect earlier lines.",
            interactive=False,
            autoscroll=True,
            elem_classes="training-log-console",
        )
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
        train.click(
            start_training,
            [image_files, name, trigger, caption, *training_controls],
            [training_status, active_job, training_log, train],
        )
        check_status.click(
            job_view,
            [check_name, name, trigger, image_files],
            [training_status, training_log, train, active_job],
        )
        required_inputs = [name, trigger, image_files]
        name.input(
            training_button_state,
            required_inputs,
            train,
            show_progress="hidden",
        )
        trigger.input(
            training_button_state,
            required_inputs,
            train,
            show_progress="hidden",
        )
        image_files.change(
            training_button_state,
            required_inputs,
            train,
            show_progress="hidden",
        )
        training_timer = gr.Timer(1.0, active=True)
        training_timer.tick(
            job_view,
            [active_job, name, trigger, image_files],
            [training_status, training_log, train, active_job],
            show_progress="hidden",
        )
        demo.load(
            job_view,
            [active_job, name, trigger, image_files],
            [training_status, training_log, train, active_job],
            show_progress="hidden",
        )
    with gr.Tab("Generate"):
        prompt = gr.Textbox(
            label="Prompt",
            info="Description of the image to generate, including the LoRA trigger word when applicable.",
        )
        lora = gr.Textbox(
            label="LoRA name (private model slug or owner/repo)",
            info="Private adapter to load. Short names resolve under the configured Space owner.",
        )
        negative = gr.Textbox(
            label="Negative prompt",
            info="Optional qualities or content to discourage in the generated image.",
        )
        with gr.Row():
            width = gr.Slider(
                256, 2048, value=1024, step=32, label="Width",
                info="Output width in pixels; larger images require more VRAM and time.",
            )
            height = gr.Slider(
                256, 2048, value=1024, step=32, label="Height",
                info="Output height in pixels; larger images require more VRAM and time.",
            )
            infer_steps = gr.Slider(
                1, 80, value=28, step=1, label="Steps",
                info="Number of denoising steps used to generate the image.",
            )
        with gr.Row():
            guidance = gr.Slider(
                0, 20, value=4, step=0.1, label="Guidance",
                info="Strength with which generation follows the text prompt.",
            )
            scale = gr.Slider(
                0, 2, value=0.8, step=0.05, label="LoRA scale",
                info="Influence of the selected LoRA on the generated image.",
            )
            seed = gr.Number(
                label="Seed (empty = random)", precision=0,
                info="Random seed for reproducible output; leave empty to choose one automatically.",
            )
        with gr.Row():
            remove_background = gr.Checkbox(
                value=True,
                label="Remove background (transparent PNG)",
                info="Segments the generated subject and stores genuine PNG alpha transparency.",
            )
            upscale_to_2k = gr.Checkbox(
                value=False,
                label="AI upscale output to 2K (Swin2SR)",
                info="Uses tiled Swin2SR neural super-resolution and preserves transparency. The longest edge becomes 2048 pixels; existing 2K images are unchanged.",
            )
        run = gr.Button("Generate", variant="primary")
        image = gr.Image(
            label="PNG result",
            format="png",
            elem_classes="transparent-result",
        )
        output_details = gr.Markdown()
        used_seed = gr.Number(
            label="Used seed",
            precision=0,
            info="Seed used for this image; reuse it to reproduce or refine the result.",
        )
        run.click(
            generate_ui,
            [
                prompt, lora, negative, width, height, infer_steps, guidance,
                scale, seed, remove_background, upscale_to_2k,
            ],
            [image, used_seed, output_details],
        )

demo.app.add_api_route("/v1/generate", generate, methods=["POST"], response_model=None)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", ssr_mode=False, css=APP_CSS)
