from app.config import Settings
from app.llm.client import LLMClient
from app.schemas.reports import DamageRegion, VascularReport


async def generate_vascular_report(
    *,
    settings: Settings,
    grade: int,
    biomarkers: dict,
    candidate_regions: list[dict],
    closeness: float,
) -> VascularReport:
    prompt = f"""Return only JSON for this schema:
{{
  "damaged_regions": [
    {{
      "x_min": int, "y_min": int, "x_max": int, "y_max": int,
      "quadrant": "NW" | "NE" | "SW" | "SE",
      "severity": "low" | "medium" | "high",
      "finding": string
    }}
  ],
  "overall_damage_score": int,
  "per_grade_severity": {{ "0": float, "1": float, "2": float, "3": float, "4": float }},
  "needs_specialist_review": bool,
  "rationale": string,
  "closeness_to_next_grade": float
}}

You are reviewing vessel-mask damage candidates from OpenCV skeletonization.
Do not invent regions outside the candidates unless the biomarkers strongly support it.
Clamp all coordinates to 0..512.

CNN grade: {grade}
Biomarkers: {biomarkers}
Candidate regions: {candidate_regions}
Closeness to next grade: {closeness}
"""
    data = None
    try:
        data = await LLMClient(settings).generate_json(prompt, vision=True)
    except Exception:
        data = None
    if data:
        try:
            return VascularReport.model_validate(data)
        except Exception:
            pass
    return _fallback_vascular_report(grade, biomarkers, candidate_regions, closeness)


def _fallback_vascular_report(
    grade: int,
    biomarkers: dict,
    candidate_regions: list[dict],
    closeness: float,
) -> VascularReport:
    clean_regions = []
    for region in candidate_regions[:8]:
        payload = {key: value for key, value in region.items() if key != "score"}
        clean_regions.append(DamageRegion.model_validate(payload))

    broken = int(biomarkers.get("num_broken_segments_estimate", 0))
    tortuosity = float(biomarkers.get("tortuosity", 1.0))
    density = float(biomarkers.get("vessel_density", 0.0))
    damage_score = min(100, int(broken * 8 + max(0, tortuosity - 1.4) * 35 + max(0, 0.12 - density) * 120))

    severity = {
        "0": max(0.05, 1.0 - damage_score / 80),
        "1": 0.18,
        "2": min(0.3, damage_score / 200),
        "3": min(0.2, max(0, damage_score - 45) / 250),
        "4": min(0.12, max(0, damage_score - 70) / 250),
    }
    total = sum(severity.values())
    severity = {key: round(value / total, 3) for key, value in severity.items()}

    return VascularReport(
        damaged_regions=clean_regions,
        overall_damage_score=damage_score,
        per_grade_severity=severity,  # type: ignore[arg-type]
        needs_specialist_review=damage_score > 60 or grade >= 3,
        rationale=(
            f"Skeleton analysis found {broken} possible broken segments with mean "
            f"tortuosity {tortuosity:.2f}; candidate boxes are limited to image-derived regions."
        ),
        closeness_to_next_grade=closeness,
    )
