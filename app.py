"""Entrypoint: mounts the generation API on the Gradio app and launches it.

Importing `ui` builds the Blocks and, through generation.py, imports `spaces`
ahead of torch. Keep that import first so ZeroGPU initializes before CUDA.
"""
from generation import generate
from ui import APP_CSS, demo

demo.app.add_api_route("/v1/generate", generate, methods=["POST"], response_model=None)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", ssr_mode=False, css=APP_CSS)
