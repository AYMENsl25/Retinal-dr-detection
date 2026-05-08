from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.preprocessing import load_fundus_image
from app.llm.clinical_explainer import build_clinical_report
from app.llm.vascular_analyst import build_vascular_report
from app.models.registry import model_registry
from app.schemas.inference import AnalyzeResponse, HealthResponse

router = APIRouter(tags=["inference"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    status = model_registry.status()
    return HealthResponse(status="ok", model_runtime=status.runtime, checkpoints=status.checkpoints)


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(image: UploadFile = File(...)) -> AnalyzeResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image file.")

    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Image file is empty.")

    try:
        fundus = load_fundus_image(contents)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    base_result = model_registry.analyze(fundus)
    clinical_report = await build_clinical_report(base_result)
    vascular_report = await build_vascular_report(base_result)

    return AnalyzeResponse(**base_result.model_dump(), clinical_report=clinical_report, vascular_report=vascular_report)

