import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from models.request_model import LLMQueryRequest
from services.local_lama_service import query_stream as local_ollama_stream
from services.cloud_ollama_service import query_stream as cloud_ollama_stream
from services.openai_service import query_stream as openai_stream
from services.gemini_service import query_stream as gemini_stream
from services.openrouter_service import query_stream as openrouter_stream
from services.nvidia_service import query_stream as nvidia_stream
from services.lm_studio_service import query_stream as lm_studio_stream

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["LLM Query"])

STREAMERS = {
    "openrouter": openrouter_stream,
    "cloud-ollama": cloud_ollama_stream,
    "lm-studio": lm_studio_stream,
    "openai": openai_stream,
    "local-ollama": local_ollama_stream,
    "nvidia": nvidia_stream,
    "gemini": gemini_stream,
}


def _stream_with_error_boundary(
    streamer,
    question: str,
    model: str,
    history: list[dict] | None = None,
    base_url: str | None = None,
):
    try:
        yield from streamer(question, model, base_url=base_url, history=history)
    except Exception:
        logger.exception("LLM stream failed")
        yield (
            "\nThe model request failed. "
            "Check API keys, model id, and upstream connectivity.\n"
        )


@router.post("/query")
def query_llm(req: LLMQueryRequest):
    streamer = STREAMERS.get(req.provider)
    if not streamer:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider: {req.provider}. Supported: {list(STREAMERS.keys())}",
        )
    history = [m.model_dump() for m in req.history] if req.history else []
    return StreamingResponse(
        _stream_with_error_boundary(
            streamer, req.question, req.model, history, req.base_url
        ),
        media_type="text/plain",
    )
