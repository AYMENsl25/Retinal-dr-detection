import base64
import json
from pathlib import Path
from typing import Any

import httpx

from app.config import settings
from app.llm.guardrails import DISCLAIMER
from app.schemas.inference import AnalyzeResponse, BaseAnalyzeResult, ChatMessage
from app.schemas.reports import ClinicalReport, DamagedRegion, VascularReport

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def _read_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def _should_call_provider(provider_ready: bool) -> bool:
    if settings.llm_mode.lower() == "mock":
        return False
    if settings.llm_mode.lower() == "provider":
        return True
    return provider_ready


def mock_clinical_report(result: BaseAnalyzeResult) -> ClinicalReport:
    labels = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]
    follow_up = [
        "Annual dilated fundus examination",
        "Repeat dilated examination in 9-12 months",
        "Repeat dilated examination in 6 months",
        "Retina specialist review within 1 month",
        "Urgent retina specialist review within 1-2 weeks",
    ][result.grade]

    return ClinicalReport(
        summary=f"{labels[result.grade]} pattern on the automated ICDR scale.",
        pathophysiology=(
            "The report combines vessel-density, tortuosity, fractal-dimension, lesion, "
            "and calibrated grading signals. This scaffold is using mock reasoning until "
            "provider keys and real checkpoints are configured."
        ),
        risk_factors=[
            "Duration of diabetes",
            "HbA1c control",
            "Hypertension",
            "Dyslipidemia",
        ],
        recommendations=[
            "Confirm findings with a dilated ophthalmic examination.",
            "Optimize glycemic, blood-pressure, and lipid control.",
            "Use OCT if macular edema is suspected.",
        ],
        follow_up_window=follow_up,
        lifestyle_advice=[
            "Smoking cessation if applicable",
            "Regular aerobic exercise as clinically appropriate",
            "Dietary pattern aligned with diabetes care plan",
        ],
        red_flags=[
            "Sudden vision loss",
            "New floaters or flashes",
            "Central distortion",
        ],
        disclaimer=DISCLAIMER,
    )


def mock_vascular_report(result: BaseAnalyzeResult) -> VascularReport:
    damage = min(100, max(0, result.grade * 22 + int(result.biomarkers.tortuosity * 8)))
    regions: list[DamagedRegion] = []
    if result.grade > 0:
        regions.append(
            DamagedRegion(
                quadrant="superior-temporal",
                severity="mild" if result.grade < 2 else "moderate",
                finding="Localized vascular irregularity in the mock vessel mask.",
            )
        )
    if result.grade >= 3:
        regions.append(
            DamagedRegion(
                quadrant="inferior-nasal",
                severity="severe",
                finding="High-grade pattern requiring specialist review.",
            )
        )

    return VascularReport(
        damaged_regions=regions,
        overall_damage_score=damage,
        per_grade_score={str(idx): round(prob * 100, 2) for idx, prob in enumerate(result.grade_probs)},
        closeness_to_next_grade=result.closeness_to_next_grade,
        needs_specialist_review=result.decision_flag != "HIGH_CONFIDENCE",
        rationale="Mock vascular analysis grounded on computed biomarkers until the vision LLM is enabled.",
    )


async def clinical_report_from_groq(result: BaseAnalyzeResult) -> ClinicalReport:
    if not _should_call_provider(settings.groq_ready):
        return mock_clinical_report(result)

    try:
        from groq import Groq

        client = Groq(api_key=settings.groq_api_key)
        payload = result.model_dump(mode="json")
        response = client.chat.completions.create(
            model=settings.clinical_llm_model,
            messages=[
                {"role": "system", "content": _read_prompt("clinical_explainer.txt")},
                {"role": "user", "content": json.dumps(payload)},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        content = response.choices[0].message.content or "{}"
        return ClinicalReport.model_validate_json(content)
    except Exception:
        return mock_clinical_report(result)


async def vascular_report_from_gemini(result: BaseAnalyzeResult) -> VascularReport:
    if not _should_call_provider(settings.gemini_ready):
        return mock_vascular_report(result)

    try:
        from google import genai
        from google.genai import types

        image_bytes = base64.b64decode(result.panels.overlay.split(",", 1)[1])
        prompt = (
            _read_prompt("vascular_damage_analyst.txt")
            + "\nBiomarkers: "
            + json.dumps(result.biomarkers.model_dump(mode="json"))
            + f"\nCNN grade: {result.grade}"
        )
        client = genai.Client(api_key=settings.gemini_api_key)
        response = client.models.generate_content(
            model=settings.vision_llm_model,
            contents=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        return VascularReport.model_validate_json(response.text or "{}")
    except Exception:
        return mock_vascular_report(result)


async def fallback_openrouter(prompt: str) -> str | None:
    if not _should_call_provider(settings.openrouter_ready):
        return None

    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": settings.fallback_llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


async def chat_from_gemini(prompt: str) -> str | None:
    if not _should_call_provider(settings.gemini_ready):
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=settings.gemini_api_key)
        response = client.models.generate_content(
            model=settings.chat_llm_model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2),
        )
        return response.text
    except Exception:
        return None


async def chat_from_groq(prompt: str) -> str | None:
    if not _should_call_provider(settings.groq_ready):
        return None

    try:
        from groq import Groq

        client = Groq(api_key=settings.groq_api_key)
        response = client.chat.completions.create(
            model=settings.clinical_llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise retinal clinical decision-support chat assistant. "
                        "Do not diagnose. Use only the provided case JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception:
        return None


async def answer_chat(messages: list[ChatMessage], case_context: AnalyzeResponse | None) -> str:
    last_user = next((message.content for message in reversed(messages) if message.role == "user"), "")
    question = last_user.lower()
    grade = case_context.grade if case_context else 1
    label = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"][grade]

    if settings.llm_mode.lower() != "mock":
        prompt = (
            _read_prompt("chat_consult.txt")
            + "\nCase context:\n"
            + (case_context.model_dump_json() if case_context else "{}")
            + "\nQuestion:\n"
            + last_user
            + "\nAnswer in 2-5 concise clinician-facing sentences. Include decision-support wording when treatment is discussed."
        )
        try:
            provider_answer = await chat_from_gemini(prompt)
            if provider_answer:
                return provider_answer
            provider_answer = await chat_from_groq(prompt)
            if provider_answer:
                return provider_answer
            provider_answer = await fallback_openrouter(prompt)
            if provider_answer:
                return provider_answer
        except Exception:
            pass

    if "follow" in question or "when" in question or "next" in question:
        window = case_context.clinical_report.follow_up_window if case_context else "9-12 months"
        return (
            f"For this automated {label} case, the suggested follow-up window is {window}. "
            "New floaters, flashes, central distortion, or sudden vision loss should trigger earlier review. "
            "This is decision support only and should be confirmed by an ophthalmologist."
        )

    if "treat" in question or "option" in question or "therapy" in question or "medication" in question:
        if grade <= 1:
            treatment = (
                "The usual focus is systemic risk-factor control, retinal follow-up, and OCT if macular edema is suspected."
            )
        elif grade == 2:
            treatment = (
                "Common next steps include ophthalmology confirmation, OCT assessment for macular edema, and closer follow-up; anti-VEGF or laser depends on confirmed edema or sight-threatening findings."
            )
        elif grade == 3:
            treatment = (
                "This pattern usually needs retina specialist review; treatment may include OCT-guided anti-VEGF, laser, or closer surveillance depending on confirmed findings."
            )
        else:
            treatment = (
                "Urgent retina review is usually needed; treatment may include panretinal photocoagulation, anti-VEGF therapy, or surgery depending on neovascular complications."
            )
        return f"{treatment} The automated output is not a diagnosis, so a licensed ophthalmologist must confirm before treatment."

    if "confidence" in question or "sure" in question or "uncertain" in question:
        confidence = case_context.calibrated_confidence if case_context else 0.0
        entropy = case_context.uncertainty.entropy if case_context else 0.0
        return (
            f"The calibrated confidence is {confidence:.0%}, with entropy {entropy:.2f}. "
            "Lower confidence, high entropy, or closeness to the next grade should be treated as a reason for specialist review."
        )

    if "biomarker" in question or "vessel" in question or "density" in question or "tortuos" in question:
        if not case_context:
            return "No case biomarkers are available yet. Run an analysis first."
        b = case_context.biomarkers
        return (
            f"The vessel biomarkers are density {b.vessel_density}, tortuosity {b.tortuosity}, "
            f"fractal dimension {b.fractal_dim}, and AVR {b.avr}. "
            "They should be interpreted alongside the fundus image, lesion masks, and clinical examination."
        )

    return (
        f"For this case, the current automated grade is {label}. "
        "Confirm the result with a licensed ophthalmologist, review OCT if macular edema is suspected, "
        "and treat this answer as decision support only."
    )
