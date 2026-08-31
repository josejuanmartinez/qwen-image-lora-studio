"""Inference side of the studio: LoRA pipeline, Swin2SR upscale, rembg cutout.

Imports `spaces` before torch so ZeroGPU is initialized first; every import path
that reaches torch goes through this module, so the UI layer takes `spaces` from here.
"""
import base64
import threading
from io import BytesIO
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

import torch
from diffusers import DiffusionPipeline
from fastapi import HTTPException
from pydantic import BaseModel, Field

from training import BASE_MODEL, hf_token, lora_repo

SWIN2SR_MODEL = "caidas/swin2SR-lightweight-x2-64"
# Segmentation models rembg can drive, best first. BiRefNet runs at 1024px against
# u2net's 320px, which is what preserves antennae, gun barrels, wings and hair.
# isnet-anime is trained on illustration rather than photos and often wins on flat
# cel-shaded art. bria-rmbg is non-commercial without a licence agreement with BRIA.
BG_REMOVAL_MODELS = (
    "birefnet-general",
    "birefnet-general-lite",
    "isnet-anime",
    "birefnet-portrait",
    "birefnet-massive",
    "isnet-general-use",
    "bria-rmbg",
    "u2net",
)
DEFAULT_BG_REMOVAL_MODEL = "birefnet-general"

pipe = None
loaded_lora = None
swin2sr_processor = None
swin2sr_model = None
rembg_sessions: dict[str, object] = {}
pipe_lock = threading.Lock()
swin2sr_lock = threading.Lock()
rembg_lock = threading.Lock()


def pipeline_has_adapter(pipeline, adapter_name: str) -> bool:
    """Check Diffusers' real adapter registry instead of trusting our slug cache."""
    try:
        adapters = pipeline.get_list_adapters()
        if isinstance(adapters, dict):
            return any(adapter_name in names for names in adapters.values())
        return adapter_name in adapters
    except (AttributeError, TypeError, ValueError):
        try:
            return adapter_name in pipeline.get_active_adapters()
        except (AttributeError, TypeError, ValueError):
            return False


def clear_pipeline_lora_state() -> None:
    global loaded_lora
    loaded_lora = None
    if pipe is None:
        return
    try:
        pipe.unload_lora_weights()
    except Exception:
        # This is best-effort recovery after an incomplete third-party load.
        # Keeping loaded_lora cleared forces another real load on the next request.
        pass


def get_pipe(lora_name: Optional[str], scale: float):
    global pipe, loaded_lora
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU Space is required for Qwen Image inference.")
    with pipe_lock:
        if pipe is None:
            pipe = DiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, token=hf_token()).to("cuda")
        repo = lora_repo(lora_name) if lora_name else None
        adapter_is_loaded = pipeline_has_adapter(pipe, "selected")
        if repo is None:
            if loaded_lora is not None or adapter_is_loaded:
                clear_pipeline_lora_state()
        elif repo != loaded_lora or not adapter_is_loaded:
            clear_pipeline_lora_state()
            try:
                pipe.load_lora_weights(repo, weight_name="adapter.safetensors", token=hf_token(), adapter_name="selected")
                pipe.set_adapters("selected", adapter_weights=scale)
            except Exception:
                clear_pipeline_lora_state()
                raise
            # Only cache the slug after Diffusers confirms the adapter exists.
            if not pipeline_has_adapter(pipe, "selected"):
                clear_pipeline_lora_state()
                raise RuntimeError(f"LoRA `{repo}` loaded without registering the `selected` adapter.")
            loaded_lora = repo
        else:
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
    scheduler: Optional[str] = None
    base_model: Optional[str] = None
    remove_background: bool = True
    background_model: str = DEFAULT_BG_REMOVAL_MODEL
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
            from transformers import Swin2SRForImageSuperResolution
            try:
                # Transformers 5 names the portable NumPy/PIL implementation explicitly.
                from transformers import Swin2SRImageProcessorPil as Swin2SRProcessor
            except ImportError:
                # Transformers 4 used this name for the same non-Torchvision processor.
                from transformers import Swin2SRImageProcessor as Swin2SRProcessor

            processor = Swin2SRProcessor.from_pretrained(SWIN2SR_MODEL)
            model = Swin2SRForImageSuperResolution.from_pretrained(SWIN2SR_MODEL)
            model.eval().to("cuda")
            swin2sr_processor = processor
            swin2sr_model = model
        return swin2sr_processor, swin2sr_model


def get_rembg_cuda_session(model_name=DEFAULT_BG_REMOVAL_MODEL):
    """Create one reusable rembg session per model and require its CUDA provider.

    Sessions are cached by name so switching models in the UI to compare them does
    not re-download or re-initialize a model that has already been loaded.
    """
    if model_name not in BG_REMOVAL_MODELS:
        raise ValueError(
            f"Unknown background removal model {model_name!r}. "
            f"Choose one of: {', '.join(BG_REMOVAL_MODELS)}."
        )
    with rembg_lock:
        if model_name not in rembg_sessions:
            try:
                import onnxruntime as ort
            except Exception as exc:
                raise RuntimeError(
                    "The rembg CUDA backend could not load. Rebuild the Space with "
                    "the rembg[gpu] requirement."
                ) from exc
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" not in providers:
                raise RuntimeError(
                    "onnxruntime-gpu loaded without CUDAExecutionProvider. "
                    f"Available providers: {', '.join(providers) or 'none'}."
                )
            from rembg import new_session

            rembg_sessions[model_name] = new_session(
                model_name,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        return rembg_sessions[model_name]


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
        if request.base_model is not None and request.base_model != BASE_MODEL:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Requested base model {request.base_model!r}, "
                    f"but the Space uses {BASE_MODEL!r}."
                ),
            )
        seed = int(request.seed if request.seed is not None else torch.seed() % (2**31 - 1))
        generator = torch.Generator(device="cuda").manual_seed(seed)
        selected_pipe = get_pipe(request.lora_name, request.lora_scale)
        scheduler = selected_pipe.scheduler.__class__.__name__
        if request.scheduler is not None and request.scheduler != scheduler:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Requested scheduler {request.scheduler!r}, "
                    f"but the Space uses {scheduler!r}."
                ),
            )
        image = selected_pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or None,
            width=request.width,
            height=request.height,
            num_inference_steps=request.steps,
            true_cfg_scale=request.guidance_scale,
            generator=generator,
        ).images[0]
        background_model = None
        if request.remove_background:
            # Qwen Image produces RGB art; use CUDA segmentation to make a genuine alpha PNG.
            # The model downloads on first use and both model and session are then reused.
            background_model = request.background_model
            if background_model not in BG_REMOVAL_MODELS:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unknown background removal model {background_model!r}. "
                        f"Choose one of: {', '.join(BG_REMOVAL_MODELS)}."
                    ),
                )
            background_session = get_rembg_cuda_session(background_model)
            from rembg import remove
            image = remove(image, session=background_session).convert("RGBA")
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
            "background_model": background_model,
            "upscaler": upscaler,
            "upscale_warning": upscale_warning,
            "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "mime_type": "image/png",
            "generation_parameters": {
                "prompt": request.prompt,
                "lora_name": request.lora_name,
                "negative_prompt": request.negative_prompt,
                "steps": request.steps,
                "guidance_scale": request.guidance_scale,
                "lora_scale": request.lora_scale,
                "seed": seed,
                "scheduler": scheduler,
                "base_model": BASE_MODEL,
                "background_model": background_model,
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
