from typing import Literal

from pydantic import BaseModel, Field


class DamagedRegion(BaseModel):
    quadrant: Literal[
        "superior-nasal",
        "superior-temporal",
        "inferior-nasal",
        "inferior-temporal",
        "macula",
        "optic-disc",
    ]
    severity: Literal["mild", "moderate", "severe"]
    finding: str


class ClinicalReport(BaseModel):
    summary: str
    pathophysiology: str
    risk_factors: list[str]
    recommendations: list[str]
    follow_up_window: str
    lifestyle_advice: list[str]
    red_flags: list[str]
    disclaimer: str = Field(default="Decision support only - not a medical diagnosis.")


class VascularReport(BaseModel):
    damaged_regions: list[DamagedRegion]
    overall_damage_score: float
    per_grade_score: dict[str, float]
    closeness_to_next_grade: float
    needs_specialist_review: bool
    rationale: str

