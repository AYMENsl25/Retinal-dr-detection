"""LLM client with cascading provider support.

Tries the primary provider first, then the fallback provider, then returns
None (which triggers deterministic fallback in the callers).

Provider chain: Gemini → Groq → deterministic
"""

import json
import logging
from typing import Any

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def generate_json(self, prompt: str, *, vision: bool = False) -> dict[str, Any] | None:
        """Try primary provider, then fallback, then return None."""
        # Try primary provider
        result = await self._call_provider(self.settings.llm_provider, prompt, vision=vision)
        if result is not None:
            return result

        # Try fallback provider
        fallback = self.settings.llm_fallback_provider
        if fallback and fallback != "none" and fallback != self.settings.llm_provider:
            logger.info("Primary LLM (%s) failed, trying fallback (%s)", self.settings.llm_provider, fallback)
            result = await self._call_provider(fallback, prompt, vision=vision)
            if result is not None:
                return result

        logger.info("All LLM providers failed, using deterministic fallback")
        return None

    async def _call_provider(self, provider: str, prompt: str, *, vision: bool = False) -> dict[str, Any] | None:
        """Dispatch to the appropriate provider method."""
        try:
            if provider == "none":
                return None
            if provider == "gemini":
                return await self._gemini_json(prompt, vision=vision)
            if provider == "ollama":
                return await self._ollama_json(prompt, vision=vision)
            if provider == "openrouter":
                model = self.settings.openrouter_vision_model if vision else self.settings.openrouter_text_model
                return await self._openai_compatible_json(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self.settings.openrouter_api_key,
                    model=model,
                    prompt=prompt,
                    extra_headers={
                        "HTTP-Referer": "http://localhost:3000",
                        "X-Title": "RetinaScope-AI",
                    },
                )
            if provider == "groq":
                return await self._openai_compatible_json(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=self.settings.groq_api_key,
                    model=self.settings.groq_text_model,
                    prompt=prompt,
                )
            if provider == "openai":
                model = self.settings.openai_vision_model if vision else self.settings.openai_text_model
                return await self._openai_compatible_json(
                    base_url="https://api.openai.com/v1",
                    api_key=self.settings.openai_api_key,
                    model=model,
                    prompt=prompt,
                )
            if provider == "anthropic":
                model = self.settings.anthropic_vision_model if vision else self.settings.anthropic_text_model
                return await self._anthropic_json(model=model, prompt=prompt)
        except Exception as exc:
            logger.warning("LLM provider '%s' failed: %s", provider, exc)
        return None

    # ------------------------------------------------------------------
    # Gemini (Google AI)
    # ------------------------------------------------------------------
    async def _gemini_json(self, prompt: str, *, vision: bool = False) -> dict[str, Any] | None:
        api_key = self.settings.gemini_api_key
        if not api_key or api_key.startswith("replace"):
            return None

        model = self.settings.gemini_vision_model if vision else self.settings.gemini_text_model
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.15,
                "responseMimeType": "application/json",
            },
        }

        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()

        data = response.json()
        # Extract text from Gemini response format
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            logger.warning("Unexpected Gemini response format: %s", data)
            return None

        return _loads_json(text)

    # ------------------------------------------------------------------
    # Ollama (local)
    # ------------------------------------------------------------------
    async def _ollama_json(self, prompt: str, *, vision: bool) -> dict[str, Any] | None:
        model = self.settings.ollama_vision_model if vision else self.settings.ollama_text_model
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.15},
        }
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
        text = response.json().get("response", "")
        return _loads_json(text)

    # ------------------------------------------------------------------
    # OpenAI-compatible (Groq, OpenRouter, OpenAI)
    # ------------------------------------------------------------------
    async def _openai_compatible_json(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        prompt: str,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        if not api_key or api_key.startswith("replace"):
            return None
        headers = {"Authorization": f"Bearer {api_key}", **(extra_headers or {})}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.15,
            "response_format": {"type": "json_object"},
        }
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return _loads_json(content)

    # ------------------------------------------------------------------
    # Anthropic (Claude)
    # ------------------------------------------------------------------
    async def _anthropic_json(self, *, model: str, prompt: str) -> dict[str, Any] | None:
        if not self.settings.anthropic_api_key or self.settings.anthropic_api_key.startswith("replace"):
            return None
        payload = {
            "model": model,
            "max_tokens": 1600,
            "temperature": 0.15,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": self.settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
        }
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
        content = response.json()["content"][0]["text"]
        return _loads_json(content)


def _loads_json(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return None
