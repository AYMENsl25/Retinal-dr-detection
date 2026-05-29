from app.config import Settings
from app.llm.client import LLMClient
from app.schemas.reports import ClinicalReport

GRADE_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]


async def generate_clinical_report(
    *,
    settings: Settings,
    grade: int,
    grade_probs: tuple[float, float, float, float, float],
    confidence: float,
    closeness: float,
    biomarkers: dict,
) -> ClinicalReport:
    prompt = f"""Return only JSON for this schema:
{{
  "summary": string,
  "pathophysiology": string,
  "risk_factors": string[],
  "recommendations": string[],
  "follow_up_window": string,
  "lifestyle_advice": string[],
  "red_flags": string[],
  "disclaimer": string
}}

You are generating a clinician-facing diabetic retinopathy decision-support report.
Ground the report in ICDR grading and conservative ophthalmology follow-up.

Case:
- grade: {grade} ({GRADE_NAMES[grade]})
- grade probabilities: {grade_probs}
- calibrated confidence: {confidence}
- closeness to next grade: {closeness}
- biomarkers: {biomarkers}

Keep it concise. Always state that this is decision support only, not a diagnosis.
"""
    data = None
    try:
        data = await LLMClient(settings).generate_json(prompt)
    except Exception:
        data = None
    if data:
        try:
            return ClinicalReport.model_validate(data)
        except Exception:
            pass
    return _fallback_clinical_report(grade, confidence, closeness, biomarkers)


def _fallback_clinical_report(
    grade: int,
    confidence: float,
    closeness: float,
    biomarkers: dict,
) -> ClinicalReport:
    windows = {
        0: "12 months",
        1: "6-12 months",
        2: "3-6 months",
        3: "1-4 weeks specialist referral",
        4: "urgent retina specialist referral",
    }
    grade_name = GRADE_NAMES[grade]
    broken = biomarkers.get("num_broken_segments_estimate", 0)
    return ClinicalReport(
        summary=(
            f"The screening result is Grade {grade} ({grade_name}) with "
            f"{confidence * 100:.1f}% calibrated confidence and "
            f"{closeness * 100:.1f}% movement toward the next grade."
        ),
        pathophysiology=(
            "Diabetic retinopathy severity is estimated from visible retinal vascular "
            "changes and the CNN grader output. Vessel density, tortuosity, and broken "
            f"segment estimates ({broken}) are treated as supportive biomarkers."
        ),
        risk_factors=[
            "Duration of diabetes and glycemic control should be reviewed.",
            "Hypertension, renal disease, and dyslipidemia may increase progression risk.",
        ],
        recommendations=[
            "Confirm the image and model output with a licensed ophthalmologist.",
            "Correlate with visual symptoms, HbA1c, blood pressure, and prior retinal history.",
        ],
        follow_up_window=windows.get(grade, "specialist review"),
        lifestyle_advice=[
            "Optimize glycemic control with the treating clinician.",
            "Monitor blood pressure, lipids, kidney status, and smoking exposure.",
        ],
        red_flags=[
            "New vision loss, floaters, flashes, eye pain, or rapid visual change require urgent care."
        ],
        disclaimer="Decision support only. Not a medical diagnosis.",
    )
