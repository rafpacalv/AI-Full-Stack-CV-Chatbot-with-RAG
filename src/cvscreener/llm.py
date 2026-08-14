"""Thin client over the Ollama REST API.

Three capabilities the rest of the project leans on:

* :meth:`OllamaClient.structured` - constrained decoding against a JSON Schema
  derived from a Pydantic model. Ollama enforces the *shape* of the output, so
  we never write defensive JSON-repair code. It does **not** enforce semantics,
  which is why callers still supply few-shot examples.
* :meth:`OllamaClient.chat_stream` - token-by-token streaming, so a ~12 s answer
  on a local 9B model still feels responsive.
* :meth:`OllamaClient.embed` - batched multilingual embeddings via bge-m3.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator, Sequence, Type, TypeVar

import httpx
from pydantic import BaseModel

from .config import settings

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Local generation on a 12B model is slow; these ceilings are generous on
# purpose so a cold model load never looks like a failure.
GENERATE_TIMEOUT = httpx.Timeout(600.0, connect=10.0)
EMBED_TIMEOUT = httpx.Timeout(180.0, connect=10.0)


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, host: str | None = None) -> None:
        self.host = (host or settings.ollama_host).rstrip("/")

    # -- introspection ----------------------------------------------------
    def is_up(self) -> bool:
        try:
            httpx.get(f"{self.host}/api/version", timeout=3.0).raise_for_status()
            return True
        except httpx.HTTPError:
            return False

    def available_models(self) -> list[str]:
        try:
            r = httpx.get(f"{self.host}/api/tags", timeout=10.0)
            r.raise_for_status()
            return sorted(m["name"] for m in r.json().get("models", []))
        except httpx.HTTPError:
            return []

    # -- generation -------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.7,
        num_predict: int = 1024,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model or settings.chat_model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }
        if system:
            payload["system"] = system
        try:
            r = httpx.post(
                f"{self.host}/api/generate", json=payload, timeout=GENERATE_TIMEOUT
            )
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(f"generate failed: {exc}") from exc
        return r.json().get("response", "")

    def structured(
        self,
        prompt: str,
        schema: Type[T],
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.2,
        num_predict: int = 2048,
        retries: int = 2,
    ) -> T:
        """Generate JSON constrained to ``schema`` and validate it into the model.

        Ollama guarantees syntactically valid JSON matching the schema, so the
        retry loop only exists for transport hiccups and the rare case where a
        value is well-formed but fails a Pydantic validator.
        """
        payload: dict[str, Any] = {
            "model": model or settings.chat_model,
            "prompt": prompt,
            "format": schema.model_json_schema(),
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }
        if system:
            payload["system"] = system

        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                r = httpx.post(
                    f"{self.host}/api/generate", json=payload, timeout=GENERATE_TIMEOUT
                )
                r.raise_for_status()
                return schema.model_validate_json(r.json()["response"])
            except Exception as exc:  # noqa: BLE001 - retried below, raised if final
                last = exc
                log.warning(
                    "structured() attempt %d/%d failed: %s", attempt + 1, retries + 1, exc
                )
                # Nudge toward a different sample on the retry.
                payload["options"]["temperature"] = min(
                    1.0, payload["options"]["temperature"] + 0.2
                )
        raise OllamaError(f"structured generation failed after {retries + 1} tries: {last}")

    def chat_stream(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        num_predict: int = 900,
    ) -> Iterator[str]:
        """Yield response tokens as they are produced."""
        payload = {
            "model": model or settings.chat_model,
            "messages": list(messages),
            "stream": True,
            "think": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }
        try:
            with httpx.stream(
                "POST", f"{self.host}/api/chat", json=payload, timeout=GENERATE_TIMEOUT
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if token := chunk.get("message", {}).get("content"):
                        yield token
                    if chunk.get("done"):
                        break
        except httpx.HTTPError as exc:
            raise OllamaError(f"chat stream failed: {exc}") from exc

    # -- embeddings -------------------------------------------------------
    def embed(self, texts: Sequence[str], *, model: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        try:
            r = httpx.post(
                f"{self.host}/api/embed",
                json={"model": model or settings.embed_model, "input": list(texts)},
                timeout=EMBED_TIMEOUT,
            )
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(f"embed failed: {exc}") from exc
        return r.json()["embeddings"]


client = OllamaClient()
