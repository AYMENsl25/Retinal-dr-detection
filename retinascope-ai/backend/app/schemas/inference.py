from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.reports import ClinicalReport, VascularReport

DRGrade = Literal[0, 1, 2, 3, 4]
DecisionFlag = Literal["HIGH_CONFIDENCE", "MEDIUM_REFER_RECOMMENDED", "REFER_SPECIALIST"]


class PanelImages(BaseModel):
    original: str
    mask: str
    heatmap: str
    overlay: str


class Biomarkers(BaseModel):
    vessel_density: float
    tortuosity: float
    fractal_dim: float
    avr: float


class Uncertainty(BaseModel):
    entropy: float
    mc_dropout_std: float


class BaseAnalyzeResult(BaseModel):
    case_id: str
    panels: PanelImages
    grade: DRGrade
    grade_probs: list[float] = Field(min_length=5, max_length=5)
    calibrated_confidence: float
    closeness_to_next_grade: float
    uncertainty: Uncertainty
    biomarkers: Biomarkers
    decision_flag: DecisionFlag


class AnalyzeResponse(BaseAnalyzeResult):
    clinical_report: ClinicalReport
    vascular_report: VascularReport


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    messages: list[ChatMessage]
    case_context: AnalyzeResponse | None = Field(default=None, alias="caseContext")


class HealthResponse(BaseModel):
    status: str
    model_runtime: str
    checkpoints: dict[str, bool]
