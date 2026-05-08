from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.llm.client import answer_chat
from app.schemas.inference import ChatRequest

router = APIRouter(tags=["chat"])


@router.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    answer = await answer_chat(request.messages, request.case_context)

    async def stream():
        for token in answer.split(" "):
            yield token + " "

    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")

