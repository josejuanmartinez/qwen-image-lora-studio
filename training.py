"""Training side of the studio: job store, AI Toolkit bootstrap, dataset captions.

Owns the job registry and the ai-toolkit subprocess. Imports nothing from the
generation or UI layers, so the dependency direction stays training <- generation <- ui.
"""
import copy
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import yaml
from huggingface_hub import HfApi

BASE_MODEL = "Qwen/Qwen-Image-2512"
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


def flatten_to_white(image):
    """Composite any transparency onto white before training sees the image.

    PIL's convert("RGB") only drops the alpha channel; it keeps whatever RGB was
    hidden underneath. Background-removed PNGs therefore bleed their original
    background straight back into the dataset. Compositing replaces it instead.
    """
    from PIL import Image
    if image.mode == "P":
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGBA")
        backdrop = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(backdrop, image)
    return image.convert("RGB")


CAPTION_SUFFIX = ".txt"


def caption_for(source: Path, typed: dict) -> str | None:
    """Caption typed into the gallery field, else a .txt sitting beside the image."""
    text = (typed or {}).get(str(source))
    if text and text.strip():
        return text.strip()
    neighbour = source.with_suffix(CAPTION_SUFFIX)
    if neighbour.is_file():
        try:
            return neighbour.read_text(encoding="utf-8").strip() or None
        except (OSError, UnicodeDecodeError):
            return None
    return None


def with_trigger(text: str, trigger_word: str) -> str:
    text = (text or "").strip()
    if not text:
        return trigger_word
    if trigger_word.lower() in text.lower():
        return text
    return f"{trigger_word}, {text}"


def load_caption_sidecars(paths) -> dict[str, str]:
    """Map lowercased file stem -> caption text for uploaded .txt files."""
    captions: dict[str, str] = {}
    for path in paths or []:
        candidate = Path(str(path))
        if candidate.suffix.lower() != CAPTION_SUFFIX:
            continue
        try:
            text = candidate.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            continue
        if text:
            captions[candidate.stem.lower()] = text
    return captions


def apply_sidecars(captions, files, sidecars, overwrite: bool):
    """Fill caption fields from uploaded .txt files matched by filename stem."""
    updated = dict(captions or {})
    sidecars = sidecars or {}
    matched = 0
    for item in files or []:
        text = sidecars.get(Path(item).stem.lower())
        if not text:
            continue
        matched += 1
        if overwrite or not (updated.get(item) or "").strip():
            updated[item] = text
    return updated, matched


def caption_progress(captions, files) -> str:
    files = list(files or [])
    captions = captions or {}
    if not files:
        return "Add images, then describe each one in the field under its thumbnail."
    done = sum(1 for item in files if (captions.get(item) or "").strip())
    if done == 0:
        return f"0 of {len(files)} images captioned - all will fall back to the shared caption below."
    if done == len(files):
        return f"All {len(files)} images captioned."
    return f"{done} of {len(files)} images captioned - the other {len(files) - done} will use the shared caption below."


def set_slot_caption(index: int):
    """Write one gallery field back into the path -> caption map."""
    def handler(text, captions, files, page):
        files = list(files or [])
        start = (max(1, int(page or 1)) - 1) * GALLERY_PAGE_SIZE
        window = files[start:start + GALLERY_PAGE_SIZE]
        updated = dict(captions or {})
        if index < len(window):
            text = (text or "").strip()
            if text:
                updated[window[index]] = text
            else:
                updated.pop(window[index], None)
        return updated, caption_progress(updated, files)
    return handler
