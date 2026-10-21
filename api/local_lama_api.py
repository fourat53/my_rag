from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from models.request_model import LLMQueryRequest
from services.local_lama_service import (
    query_stream,
)

router = APIRouter(prefix="/llama", tags=["LLAMA"])


@router.post("/query")
def query_text(req: LLMQueryRequest):
    history = [m.model_dump() for m in req.history] if req.history else []
    return StreamingResponse(
        query_stream(
            req.question, req.model, base_url=req.base_url, history=history
        ),
        media_type="text/plain",
    )
