from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "RetinaScope-AI"
    app_version: str = Field(default="2.0.0-dev", alias="APP_VERSION")
    environment: str = "development"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    vessel_model_path: Path = Path("checkpoints/vessel_unet.pth")
    vessel_preprocess_path: Path = Path("checkpoints/preprocess.json")
    grader_model_path: Path = Path("checkpoints/grader_cnn.pth")
    grader_preprocess_path: Path = Path("checkpoints/preprocess.json")
    model_image_size: int = 512
    model_device: str = "auto"

    # Primary LLM provider
    llm_provider: Literal["none", "gemini", "ollama", "openrouter", "groq", "openai", "anthropic"] = "gemini"
    # Fallback LLM provider (tried if primary fails)
    llm_fallback_provider: Literal["none", "gemini", "ollama", "openrouter", "groq", "openai", "anthropic"] = "groq"

    # Gemini
    gemini_api_key: str | None = None
    gemini_text_model: str = "gemini-2.0-flash-lite"
    gemini_vision_model: str = "gemini-2.0-flash-lite"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_text_model: str = "llama3.1:8b"
    ollama_vision_model: str = "llama3.2-vision:11b"

    # OpenRouter
    openrouter_api_key: str | None = None
    openrouter_text_model: str = "openrouter/free"
    openrouter_vision_model: str = "openrouter/free"

    # Groq
    groq_api_key: str | None = None
    groq_text_model: str = "llama-3.3-70b-versatile"

    # OpenAI
    openai_api_key: str | None = None
    openai_text_model: str = "gpt-4.1-mini"
    openai_vision_model: str = "gpt-4.1-mini"

    # Anthropic
    anthropic_api_key: str | None = None
    anthropic_text_model: str = "claude-sonnet-4-5"
    anthropic_vision_model: str = "claude-sonnet-4-5"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
