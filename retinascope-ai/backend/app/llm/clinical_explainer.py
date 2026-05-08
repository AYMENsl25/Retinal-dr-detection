from app.llm.client import clinical_report_from_groq
from app.llm.guardrails import enforce_clinical_guardrails
from app.schemas.inference import BaseAnalyzeResult
from app.schemas.reports import ClinicalReport


async def build_clinical_report(result: BaseAnalyzeResult) -> ClinicalReport:
    report = await clinical_report_from_groq(result)
    return enforce_clinical_guardrails(report)

