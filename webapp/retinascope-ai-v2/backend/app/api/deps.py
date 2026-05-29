from app.config import Settings, get_settings
from app.models.registry import ModelRegistry, get_model_registry


def settings_dep() -> Settings:
    return get_settings()


def registry_dep() -> ModelRegistry:
    return get_model_registry()
