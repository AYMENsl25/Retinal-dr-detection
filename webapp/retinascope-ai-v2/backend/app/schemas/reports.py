from typing import Literal

from pydantic import BaseModel, Field, field_validator

Severity = Literal["low", "medium", "high"]
Quadrant = Literal["NW", "NE", "SW", "SE"]


class DamageRegion(BaseModel):
    x_min: int = Field(ge=0, le=512)
    y_min: int = Field(ge=0, le=512)
    x_max: int = Field(ge=0, le=512)
    y_max: int = Field(ge=0, le=512)
    quadrant: Quadrant
    severity: Severity
    finding: str = Field(max_length=80)

    @field_validator("x_max")
    @classmethod
    def x_order(cls, value: int, info):
        x_min = info.data.get("x_min")
        if x_min is not None and value < x_min:
            raise ValueError("x_max must be >= x_min")
        return value

    @field_validator("y_max")
    @classmethod
    def y_order(cls, value: int, info):
        y_min = info.data.get("y_min")
        if y_min is not None and value < y_min:
            raise ValueError("y_max must be >= y_min")
        return value


class ClinicalReport(BaseModel):
    summary: str
    pathophysiology: str
    risk_factors: list[str]
    recommendations: list[str]
    follow_up_window: str
    lifestyle_advice: list[str]
    red_flags: list[str]
    disclaimer: str


class VascularReport(BaseModel):
    damaged_regions: list[DamageRegion] = Field(default_factory=list, max_length=8)
    overall_damage_score: int = Field(ge=0, le=100)
    per_grade_severity: dict[Literal["0", "1", "2", "3", "4"], float]
    needs_specialist_review: bool
    rationale: str
    closeness_to_next_grade: float = Field(ge=0.0, le=1.0)
