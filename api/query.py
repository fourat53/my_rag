import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from models.request import LLMQueryRequest
from services.query import (
    lm_studio_stream,
    local_ollama_stream,
    cloud_ollama_stream,
    openrouter_stream,
    nvidia_stream,
    gemini_stream,
    openai_stream,
)

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


def stream_response(
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
        stream_response(streamer, req.question, req.model, history, req.base_url),
        media_type="text/plain",
    )
