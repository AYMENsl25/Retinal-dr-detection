from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent

PLACEHOLDER_PREFIXES = (
    "replace-with",
    "your-",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    app_name: str = "RetinaScope-AI"
    api_v1_prefix: str = "/api/v1"
    frontend_origin: str = "http://localhost:3000"

    model_device: str = "auto"
    checkpoint_dir: str = "backend/checkpoints"
    vessel_checkpoint: str = "vessel_unet.pth"
    lesion_checkpoint: str = "lesion_unet.pth"
    grader_checkpoint: str = "grader_cnn.pth"
    preprocess_config: str = "preprocess.json"

    llm_mode: str = "auto"
    clinical_llm_provider: str = "groq"
    clinical_llm_model: str = "llama-3.3-70b-versatile"
    vision_llm_provider: str = "gemini"
    vision_llm_model: str = "gemini-3.1-flash-lite"
    chat_llm_provider: str = "gemini"
    chat_llm_model: str = "gemini-3.1-flash-lite"
    fallback_llm_provider: str = "openrouter"
    fallback_llm_model: str = "openrouter/free"

    groq_api_key: str = ""
    gemini_api_key: str = ""
    openrouter_api_key: str = ""

    def resolved_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return REPO_ROOT / path

    @property
    def checkpoint_path(self) -> Path:
        return self.resolved_path(self.checkpoint_dir)

    def checkpoint_file(self, filename: str) -> Path:
        return self.checkpoint_path / filename

    def has_secret(self, value: str) -> bool:
        lowered = value.strip().lower()
        if not lowered:
            return False
        return not any(lowered.startswith(prefix) for prefix in PLACEHOLDER_PREFIXES)

    @property
    def groq_ready(self) -> bool:
        return self.has_secret(self.groq_api_key)

    @property
    def gemini_ready(self) -> bool:
        return self.has_secret(self.gemini_api_key)

    @property
    def openrouter_ready(self) -> bool:
        return self.has_secret(self.openrouter_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
