from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from models.request_model import (
    AddTextRequest,
    DeleteTextRequest,
    QueryRequest,
    SearchRequest,
)
from models.response_model import (
    AddTextResponse,
    DeleteTextResponse,
    QueryResponse,
    SearchResponse,
    ChunkSimilarityItem,
)
from services.local_lama_service import (
    add_text_to_db,
    delete_text_from_db,
    query_db,
    query_db_stream,
    similarity_search_chunks,
)
import time

router = APIRouter(prefix="/llama", tags=["LLAMA"])


@router.post("/add", response_model=AddTextResponse)
def add_text(req: AddTextRequest):
    added_chunks, added_count = add_text_to_db(req.index_id, req.text)
    return AddTextResponse(
        status="success",
        added_chunks=added_chunks,
        added_count=added_count,
    )


@router.delete("/delete", response_model=DeleteTextResponse)
def delete_text(req: DeleteTextRequest):
    deleted_chunks, deleted_count = delete_text_from_db(req.index_id)
    return DeleteTextResponse(
        status="success",
        deleted_chunks=deleted_chunks,
        deleted_count=deleted_count,
    )


@router.post("/query")
def query_text(req: QueryRequest):
    return StreamingResponse(query_db_stream(req.question), media_type="text/plain")


@router.post("/search", response_model=SearchResponse)
def similarity_search(req: SearchRequest):
    start_time = time.time()
    results = similarity_search_chunks(req.query, k=req.top_k)
    chunks = [
        ChunkSimilarityItem(content=content, score=score) for content, score in results
    ]
    latency = time.time() - start_time
    return SearchResponse(
        latency=latency,
        total_count=len(chunks),
        chunks=chunks,
    )
