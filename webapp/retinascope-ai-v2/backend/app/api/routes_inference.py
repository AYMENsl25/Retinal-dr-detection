from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, Depends, File, UploadFile

from app.api.deps import registry_dep, settings_dep
from app.config import Settings
from app.core.metrics import compute_vessel_biomarkers
from app.core.vessel_analysis import analyze_vessel_damage
from app.core.visualizer import build_panels, build_zoom_crops
from app.llm.clinical_explainer import generate_clinical_report
from app.llm.vascular_analyst import generate_vascular_report
from app.models.registry import ModelRegistry
from app.schemas.inference import AnalyzeResponse
from app.schemas.reports import DamageRegion
from app.utils.image_io import decode_image_bytes

router = APIRouter(prefix="/api/v1", tags=["inference"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    settings: Settings = Depends(settings_dep),
    registry: ModelRegistry = Depends(registry_dep),
) -> AnalyzeResponse:
    image_bytes = await file.read()
    image = decode_image_bytes(image_bytes, size=None)

    prediction = registry.predict(image)
    prepared_image = prediction.prepared_image
    vessel_analysis = analyze_vessel_damage(prediction.clean_mask, size=settings.model_image_size)
    biomarkers = compute_vessel_biomarkers(
        prediction.clean_mask,
        tortuosity=vessel_analysis["mean_tortuosity"],
        broken_segments=vessel_analysis["broken_segments_estimate"],
    )

    grade_probs = prediction.grade_probs
    grade = int(np.argmax(np.array(grade_probs)))
    confidence = float(grade_probs[grade])
    next_grade = min(grade + 1, 4)
    closeness = float(grade_probs[next_grade] / (grade_probs[grade] + grade_probs[next_grade] + 1e-6))

    vascular_report = await generate_vascular_report(
        settings=settings,
        grade=grade,
        biomarkers=biomarkers,
        candidate_regions=vessel_analysis["candidate_regions"],
        closeness=round(closeness, 3),
    )
    clinical_report = await generate_clinical_report(
        settings=settings,
        grade=grade,
        grade_probs=grade_probs,
        confidence=confidence,
        closeness=round(closeness, 3),
        biomarkers=biomarkers,
    )

    regions: list[DamageRegion] = vascular_report.damaged_regions
    panels = build_panels(prepared_image, prediction.prob_map, prediction.clean_mask, regions)
    zoom_crops = build_zoom_crops(prediction.clean_mask, regions)

    decision_flag = "HIGH_CONFIDENCE"
    if grade >= 3 or vascular_report.needs_specialist_review:
        decision_flag = "HIGH_CONCERN"
    elif prediction.entropy > 0.6 or grade >= 1:
        decision_flag = "MEDIUM"

    return AnalyzeResponse(
        case_id=f"RS-{uuid4().hex[:8].upper()}",
        created_at=datetime.now(UTC).isoformat(),
        panels=panels,
        damage_zoom_crops=zoom_crops,
        grade=grade,  # type: ignore[arg-type]
        grade_probs=grade_probs,
        calibrated_confidence=round(confidence, 3),
        closeness_to_next_grade=round(closeness, 3),
        uncertainty={
            "entropy": prediction.entropy,
            "mc_dropout_std": prediction.mc_dropout_std,
        },
        biomarkers=biomarkers,
        decision_flag=decision_flag,  # type: ignore[arg-type]
        clinical_report=clinical_report,
        vascular_report=vascular_report,
    )
