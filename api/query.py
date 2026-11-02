import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_openai.chat_models.base import OpenAIModelNotFoundError
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
    api_key: str | None = None,
):
    try:
        yield from streamer(
            question,
            model,
            base_url=base_url,
            history=history,
            api_key=api_key,
        )
    except OpenAIModelNotFoundError as error:
        logger.warning("LLM model not found on provider upstream: %s", error)
        yield (
            "\nThe selected model is not available for the configured API key. "
            "Pick a different model in the model selector.\n"
        )
    except Exception:
        logger.exception("LLM stream failed")
        yield (
            "\nThe model request failed. "
            "Check API keys, model id, and upstream connectivity.\n"
        )


def stream_llm(
    provider: str,
    model: str,
    question: str,
    history: list[dict] | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> StreamingResponse:
    streamer = STREAMERS.get(provider)
    if not streamer:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider: {provider}. Supported: {list(STREAMERS.keys())}",
        )
    return StreamingResponse(
        stream_response(streamer, question, model, history, base_url, api_key),
        media_type="text/plain",
    )


@router.post("/query")
def query_llm(req: LLMQueryRequest):
    return stream_llm(
        req.provider,
        req.model,
        req.question,
        history=[m.model_dump() for m in req.history] if req.history else [],
        base_url=req.base_url,
        api_key=req.api_key,
    )
