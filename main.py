from __future__ import annotations

import os

import gradio as gr
import uvicorn
from fastapi import FastAPI

from api import router
from chat_ui import build_chat_ui


def create_app() -> FastAPI:
    """Create the FastAPI application with REST endpoints and Gradio UI attached."""
    fastapi_app = FastAPI(title="Google ADK Data Science Agent")
    fastapi_app.include_router(router)
    return gr.mount_gradio_app(fastapi_app, build_chat_ui(), path="/")


app: FastAPI = create_app()


def main() -> None:
    """Run a development server for the combined FastAPI + Gradio app."""
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
