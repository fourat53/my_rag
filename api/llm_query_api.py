import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from models.request_model import LLMQueryRequest
from services.local_lama_service import query_db_stream as ollama_stream
from services.cloud_ollama_service import query_db_stream as cloud_ollama_stream
from services.openai_service import query_db_stream as openai_stream
from services.gemini_service import query_db_stream as gemini_stream
from services.openrouter_service import query_db_stream as openrouter_stream
from services.nvidia_service import query_db_stream as nvidia_stream

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["LLM Query"])

STREAMERS = {
    "local-ollama": ollama_stream,
    "cloud-ollama": cloud_ollama_stream,
    "openai": openai_stream,
    "gemini": gemini_stream,
    "openrouter": openrouter_stream,
    "nvidia": nvidia_stream,
}


def _stream_with_error_boundary(streamer, question: str, model: str):
    try:
        yield from streamer(question, model)
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
    return StreamingResponse(
        _stream_with_error_boundary(streamer, req.question, req.model),
        media_type="text/plain",
    )
