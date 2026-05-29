from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.deps import settings_dep
from app.config import Settings
from app.llm.client import LLMClient

router = APIRouter(prefix="/api/v1", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    case_context: dict | None = None


@router.post("/chat")
async def chat(payload: ChatRequest, settings: Settings = Depends(settings_dep)) -> dict:
    prompt = f"""Answer in under 140 words using only this retina screening case context.
End with: Decision support only. Not a medical diagnosis.

Context:
{payload.case_context or {}}

Question:
{payload.message}

Return JSON: {{ "answer": string }}
"""
    try:
        data = await LLMClient(settings).generate_json(prompt)
    except Exception:
        data = None
    if data and isinstance(data.get("answer"), str):
        return {"answer": data["answer"]}
    return {
        "answer": (
            "I can explain the current case grade, confidence, vascular biomarkers, "
            "and follow-up considerations once the structured report is available. "
            "Decision support only. Not a medical diagnosis."
        )
    }
