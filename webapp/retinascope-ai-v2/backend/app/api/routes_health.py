from fastapi import APIRouter

from app.config import get_settings
from app.models.registry import get_model_registry

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health")
def health() -> dict:
    settings = get_settings()
    registry = get_model_registry()
    return {
        "status": "ok",
        "version": settings.app_version,
        "environment": settings.environment,
        "llm_provider": settings.llm_provider,
        "models": registry.status(),
    }
