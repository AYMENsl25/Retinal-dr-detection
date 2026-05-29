from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.reports import ClinicalReport, VascularReport

DecisionFlag = Literal["HIGH_CONFIDENCE", "MEDIUM", "HIGH_CONCERN"]


class Biomarkers(BaseModel):
    vessel_density: float
    tortuosity: float
    fractal_dim: float
    avr: float
    num_vessel_components: int
    num_broken_segments_estimate: int
    quadrant_density: dict[Literal["NW", "NE", "SW", "SE"], float]


class Panels(BaseModel):
    original: str
    heatmap: str
    vessel_clean: str
    vessel_annotated: str
    overlay: str


class ZoomCrop(BaseModel):
    image: str
    finding: str
    severity: Literal["low", "medium", "high"]
    quadrant: Literal["NW", "NE", "SW", "SE"]


class Uncertainty(BaseModel):
    entropy: float
    mc_dropout_std: float


class AnalyzeResponse(BaseModel):
    case_id: str
    created_at: str
    panels: Panels
    damage_zoom_crops: list[ZoomCrop] = Field(default_factory=list)
    grade: Literal[0, 1, 2, 3, 4]
    grade_probs: tuple[float, float, float, float, float]
    calibrated_confidence: float
    closeness_to_next_grade: float
    uncertainty: Uncertainty
    biomarkers: Biomarkers
    decision_flag: DecisionFlag
    clinical_report: ClinicalReport
    vascular_report: VascularReport
