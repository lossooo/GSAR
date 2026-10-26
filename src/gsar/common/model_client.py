"""OpenAI-compatible model clients configured only through environment variables."""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from typing import Any

import numpy as np
from openai import OpenAI
from PIL import Image


def _provider_headers() -> dict[str, str] | None:
    provider = os.getenv("GSAR_MODEL_PROVIDER_ID", "").strip()
    return {"X-Model-Provider-Id": provider} if provider else None


OPENAI_CONFIG: dict[str, Any] = {
    "base_url": os.getenv("GSAR_BASE_URL", "https://api.openai.com/v1"),
    "api_key": os.getenv("GSAR_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
    "model_name": os.getenv("GSAR_MODEL", "gpt-4o"),
}
if _provider_headers():
    OPENAI_CONFIG["default_headers"] = _provider_headers()


def configure_model(*, model: str | None = None) -> None:
    """Apply a runtime model override before evaluators are instantiated."""
    if model:
        OPENAI_CONFIG["model_name"] = model


def _client(model_name: str | None = None) -> tuple[OpenAI, str]:
    api_key = OPENAI_CONFIG["api_key"]
    if not api_key:
        raise RuntimeError("Set GSAR_API_KEY (or OPENAI_API_KEY) before calling a model.")
    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "base_url": OPENAI_CONFIG["base_url"],
    }
    if OPENAI_CONFIG.get("default_headers"):
        kwargs["default_headers"] = OPENAI_CONFIG["default_headers"]
    return OpenAI(**kwargs), model_name or OPENAI_CONFIG["model_name"]


def create_openai_client() -> OpenAI:
    """Create a configured OpenAI-compatible client."""
    return _client()[0]


class MLLM_Model:
    """Small compatibility wrapper used by the original synthesis code."""

    def __init__(self, model_name: str | None = None):
        self.client, self.model_name = _client(model_name)
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _encode_path(image_path: str) -> str:
        path = os.path.expanduser(image_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        suffix = os.path.splitext(path)[1].lower()
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}.get(suffix, "application/octet-stream")
        with open(path, "rb") as handle:
            payload = base64.b64encode(handle.read()).decode("utf-8")
        return f"data:{mime};base64,{payload}"

    @staticmethod
    def _encode_array(image: np.ndarray) -> str:
        buffer = io.BytesIO()
        Image.fromarray(image).convert("RGB").save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_model_response(
        self,
        prompt: str,
        image_paths: list[str] | None = None,
        images: list[np.ndarray] | None = None,
        max_tokens: int = 2048,
        **kwargs: Any,
    ):
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_path in image_paths or []:
            content.append({"type": "image_url", "image_url": {"url": self._encode_path(image_path)}})
        for image in images or []:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._encode_array(image)}"}})

        attempts = int(os.getenv("GSAR_API_RETRIES", "5"))
        for attempt in range(1, attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=max_tokens,
                    stream=False,
                    **kwargs,
                )
                return response.choices[0].message.content, response
            except Exception:
                if attempt == attempts:
                    raise
                self.logger.warning("Model request failed; retrying (%s/%s)", attempt, attempts)
                time.sleep(5)
