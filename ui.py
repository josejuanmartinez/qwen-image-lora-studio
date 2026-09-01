"""Gradio layer: event handlers and the Blocks layout.

`demo` is built at import time, matching the original module-level construction.
`spaces` comes from generation so the ZeroGPU import always precedes torch.
"""
import base64
import copy
import threading
import time
from io import BytesIO
from pathlib import Path

import gradio as gr
import yaml

from generation import (
    BG_REMOVAL_MODELS,
    DEFAULT_BG_REMOVAL_MODEL,
    GenerateRequest,
    generate,
    spaces,
)
from training import (
    ACTIVE_JOB_STATUSES,
    GALLERY_PAGE_SIZE,
    JOBS,
    LATEST_JOB_FILE,
    TRAINING_PARAM_NAMES,
    apply_sidecars,
    caption_for,
    caption_progress,
    flatten_to_white,
    job_log_path,
    jobs,
    jobs_lock,
    latest_job_name,
    load_caption_sidecars,
    load_job_metadata,
    lora_repo,
    persist_job_metadata,
    qwen_config,
    read_job_log,
    run_training,
    set_job_status,
    set_slot_caption,
    slug,
    with_trigger,
)

# Presentation is isolated in style.css; this module keeps only layout and behaviour.
APP_CSS = (Path(__file__).parent / "style.css").read_text(encoding="utf-8")


def start_training(files, lora_name: str, trigger_word: str, caption: str, image_caption_map, *param_values):
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
    shared_caption = caption.strip() or trigger_word
    typed_captions = dict(image_caption_map or {})
    flattened_count = 0
    sidecar_count = 0
    for index, path in enumerate(files):
        source = Path(path[0] if isinstance(path, (tuple, list)) else path)
        target = image_dir / f"{index:04d}.png"
        try:
            with Image.open(source) as uploaded:
                oriented = ImageOps.exif_transpose(uploaded)
                if oriented.mode in ("RGBA", "LA") or (oriented.mode == "P" and "transparency" in oriented.info):
                    flattened_count += 1
                flatten_to_white(oriented).save(target, format="PNG")
        except Exception as exc:
            set_job_status(run_name, "failed", f"ERROR: Could not read {source.name}: {exc}")
            raise gr.Error(f"Could not read {source.name}: {exc}") from exc
        per_image = caption_for(source, typed_captions)
        if per_image:
            sidecar_count += 1
        target.with_suffix(".txt").write_text(
            with_trigger(per_image or shared_caption, trigger_word), encoding="utf-8"
        )
    notes = [f"Queued {len(files)} training images."]
    if flattened_count:
        notes.append(f"Composited transparency onto white for {flattened_count} image(s).")
    notes.append(
        f"Used per-image captions for {sidecar_count} of {len(files)} images."
        if sidecar_count
        else "No per-image captions found; all images share one caption."
    )
    set_job_status(run_name, "queued", " ".join(notes))
    threading.Thread(target=run_training, args=(run_name, image_dir, config), daemon=True).start()
    return (
        f"### ⏳ Queued `{run_name}`\nIt will publish privately to `{lora_repo(run_name)}` when complete.",
        run_name,
        read_job_log(run_name),
        gr.update(interactive=False, value="Training in progress…"),
    )


def gallery_page(files, page: int = 1, captions=None):
    files = list(files or [])
    page_count = max(1, (len(files) + GALLERY_PAGE_SIZE - 1) // GALLERY_PAGE_SIZE)
    page = min(max(1, int(page or 1)), page_count)
    start = (page - 1) * GALLERY_PAGE_SIZE
    window = files[start:start + GALLERY_PAGE_SIZE]
    captions = captions or {}
    slots = []
    for offset in range(GALLERY_PAGE_SIZE):
        if offset < len(window):
            path = window[offset]
            slots.append(gr.update(visible=True))
            slots.append(gr.update(value=path, label=Path(path).name))
            slots.append(gr.update(value=captions.get(path, "")))
        else:
            slots.append(gr.update(visible=False))
            slots.append(gr.update(value=None, label=""))
            slots.append(gr.update(value=""))
    summary = f"Page {page} of {page_count} · {len(files)} image{'s' if len(files) != 1 else ''}"
    return (
        *slots,
        page,
        summary,
        gr.update(interactive=page > 1),
        gr.update(interactive=page < page_count),
    )


def add_gallery_files(new_files, existing_files, captions, sidecars):
    accumulated = list(existing_files or [])
    for path in new_files or []:
        path = str(path)
        if path not in accumulated:
            accumulated.append(path)
    updated, _ = apply_sidecars(captions, accumulated, sidecars, overwrite=False)
    return (
        accumulated,
        updated,
        caption_progress(updated, accumulated),
        *gallery_page(accumulated, 1, updated),
    )


def add_caption_files(new_files, captions, files, sidecars, page):
    """Load .txt captions named after the images and drop them into the fields."""
    files = list(files or [])
    loaded = load_caption_sidecars(new_files)
    merged = dict(sidecars or {})
    merged.update(loaded)
    updated, matched = apply_sidecars(captions, files, loaded, overwrite=True)
    if not files:
        note = (
            f"Loaded {len(loaded)} caption file(s). "
            "They will fill in automatically as you add the matching images."
        )
    else:
        note = f"Loaded {len(loaded)} caption file(s); {matched} matched an image by name."
        stems = {Path(item).stem.lower() for item in files}
        unmatched = sorted(stem for stem in loaded if stem not in stems)
        if unmatched:
            shown = ", ".join(unmatched[:5]) + ("…" if len(unmatched) > 5 else "")
            note += f" No image named: {shown}."
    return (
        updated,
        merged,
        f"{note} {caption_progress(updated, files)}",
        *gallery_page(files, page, updated),
    )


def change_gallery_page(files, page: int, delta: int, captions):
    return gallery_page(files, int(page or 1) + delta, captions)


def clear_gallery():
    return [], {}, {}, caption_progress({}, []), *gallery_page([], 1, {})


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


@spaces.GPU(duration=120)
def generate_ui(prompt, lora_name, negative_prompt, width, height, steps, guidance, scale, seed, remove_background, background_model, upscale_to_2k):
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
        background_model=background_model or DEFAULT_BG_REMOVAL_MODEL,
        upscale_to_2k=bool(upscale_to_2k),
    ))
    from PIL import Image
    transparency = "transparent PNG" if result["transparent"] else "opaque PNG"
    # Name the segmentation model in the caption so A/B comparisons stay attributable.
    cutout = f" · {result['background_model']}" if result["background_model"] else ""
    upscaler = f" · {result['upscaler']}" if result["upscaler"] else ""
    warning = f"\n\n⚠️ {result['upscale_warning']}" if result["upscale_warning"] else ""
    details = f"**{result['width']} × {result['height']}** · {transparency}{cutout} · seed `{result['seed']}`{upscaler}{warning}"
    return Image.open(BytesIO(base64.b64decode(result["image_base64"]))), result["seed"], details


with gr.Blocks(title="Qwen Image LoRA Studio") as demo:
    gr.Markdown("# Qwen Image LoRA Studio\nPrivate Qwen-Image-2512 LoRA training via ostris AI Toolkit.")
    with gr.Tab("Train"):
        image_files = gr.State([])
        image_captions = gr.State({})
        caption_sidecars = gr.State({})
        gallery_page_number = gr.State(1)
        with gr.Row():
            upload = gr.UploadButton(
                "Add training images",
                file_count="multiple",
                file_types=["image"],
                type="filepath",
                variant="primary",
            )
            caption_upload = gr.UploadButton(
                "Add caption .txt files",
                file_count="multiple",
                file_types=[".txt"],
                type="filepath",
            )
            clear_images = gr.Button("Clear images")
        caption_status = gr.Markdown(
            "Add images, then describe each one in the field under its thumbnail. "
            "You can also upload .txt files named after the images (cammy.txt -> cammy.png)."
        )
        slot_columns, slot_images, slot_captions = [], [], []
        for _row in range(GALLERY_PAGE_SIZE // 4):
            with gr.Row():
                for _col in range(4):
                    with gr.Column(visible=False, min_width=180) as slot_column:
                        slot_image = gr.Image(
                            interactive=False,
                            height=220,
                        )
                        slot_caption = gr.Textbox(
                            placeholder="Describe this image…",
                            lines=2,
                            max_lines=4,
                            show_label=False,
                            container=False,
                        )
                    slot_columns.append(slot_column)
                    slot_images.append(slot_image)
                    slot_captions.append(slot_caption)
        with gr.Row():
            previous_page = gr.Button("Previous", interactive=False)
            gallery_summary = gr.Markdown("Page 1 of 1 · 0 images")
            next_page = gr.Button("Next", interactive=False)
        slot_outputs = []
        for slot_column, slot_image, slot_caption in zip(slot_columns, slot_images, slot_captions):
            slot_outputs.extend([slot_column, slot_image, slot_caption])
        gallery_outputs = [
            *slot_outputs,
            gallery_page_number,
            gallery_summary,
            previous_page,
            next_page,
        ]
        upload.upload(
            add_gallery_files,
            [upload, image_files, image_captions, caption_sidecars],
            [image_files, image_captions, caption_status, *gallery_outputs],
        )
        caption_upload.upload(
            add_caption_files,
            [caption_upload, image_captions, image_files, caption_sidecars, gallery_page_number],
            [image_captions, caption_sidecars, caption_status, *gallery_outputs],
        )
        for slot_index, slot_caption in enumerate(slot_captions):
            slot_caption.change(
                set_slot_caption(slot_index),
                [slot_caption, image_captions, image_files, gallery_page_number],
                [image_captions, caption_status],
            )
        previous_page.click(
            lambda current_files, current_page, captions: change_gallery_page(
                current_files, current_page, -1, captions
            ),
            [image_files, gallery_page_number, image_captions],
            gallery_outputs,
        )
        next_page.click(
            lambda current_files, current_page, captions: change_gallery_page(
                current_files, current_page, 1, captions
            ),
            [image_files, gallery_page_number, image_captions],
            gallery_outputs,
        )
        clear_images.click(
            clear_gallery,
            outputs=[
                image_files, image_captions, caption_sidecars,
                caption_status, *gallery_outputs,
            ],
        )

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
            [image_files, name, trigger, caption, image_captions, *training_controls],
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
                0, 2, value=1.25, step=0.05, label="LoRA scale",
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
        with gr.Row():
            background_model = gr.Dropdown(
                choices=list(BG_REMOVAL_MODELS),
                value=DEFAULT_BG_REMOVAL_MODEL,
                label="Background removal model",
                info=(
                    "birefnet-general segments at 1024px and keeps thin details such as "
                    "antennae, barrels and hair; isnet-anime is tuned for flat illustration; "
                    "u2net is the fastest but lowest quality. Each model downloads once on "
                    "first use. bria-rmbg is non-commercial without a BRIA licence."
                ),
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
                scale, seed, remove_background, background_model, upscale_to_2k,
            ],
            [image, used_seed, output_details],
        )
