"""Entrypoint: serves the generation API and the Gradio UI from one FastAPI app.

Importing `generation` pulls in `spaces` ahead of torch, and importing `ui` builds the Blocks.
Keep those two imports first so ZeroGPU initializes before CUDA.

The API route cannot be hung off `demo.app`: `Blocks.launch()` rebuilds `Blocks.app` from
scratch, so a route registered beforehand is thrown away and `POST /v1/generate` answers a JSON
404 while the UI itself serves normally. Owning the FastAPI app and mounting the Blocks onto it
with `mount_gradio_app` is the supported way to keep both on one port.
"""
from generation import generate
from ui import APP_CSS, demo

import os

import gradio as gr
import uvicorn
from fastapi import FastAPI

app = FastAPI()
app.add_api_route("/v1/generate", generate, methods=["POST"], response_model=None)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    uvicorn.run(
        gr.mount_gradio_app(app, demo, path="/", css=APP_CSS, ssr_mode=False),
        host="0.0.0.0",
        # Spaces publishes 7860; honour an override rather than assuming it.
        port=int(os.getenv("GRADIO_SERVER_PORT") or os.getenv("PORT") or 7860),
    )
