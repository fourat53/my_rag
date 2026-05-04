from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from models.response_model import (
    AddTextResponse,
    DeleteTextResponse,
    QueryResponse,
    SearchResponse,
    ChunkSimilarityItem,
)
from models.request_model import (
    LLMQueryRequest
)
from services.local_lama_service import (
    query_db_stream,
)
import time

router = APIRouter(prefix="/llama", tags=["LLAMA"])

@router.post("/query")
def query_text(req: LLMQueryRequest):
    return StreamingResponse(query_db_stream(req.question, req.model), media_type="text/plain")
