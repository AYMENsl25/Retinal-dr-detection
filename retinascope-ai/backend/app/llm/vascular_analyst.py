from app.llm.client import vascular_report_from_gemini
from app.schemas.inference import BaseAnalyzeResult
from app.schemas.reports import VascularReport


async def build_vascular_report(result: BaseAnalyzeResult) -> VascularReport:
    return await vascular_report_from_gemini(result)

