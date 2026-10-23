import os
import time
from urllib.parse import unquote, urlparse

import requests
from fastapi import APIRouter, HTTPException
from models.request_model import (
    AddTextRequest,
    DeleteTextRequest,
    QueryRequest,
    SearchRequest,
    UploadUrlRequest,
)
from models.response_model import (
    AddTextResponse,
    DeleteTextResponse,
    QueryResponse,
    SearchResponse,
    ChunkSimilarityItem,
    UploadTextResponse,
)
from services.rag_service import (
    add_text_to_db,
    delete_text_from_db,
    extract_text_from_file,
    query_db,
    similarity_search_chunks,
)

router = APIRouter(prefix="/rag", tags=["RAG"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}
MAX_UPLOAD_SIZE_BYTES = 15 * 1024 * 1024
UPLOADTHING_HOST_SUFFIXES = (".ufs.sh", ".utfs.io", "uploadthing.com")


def is_allowed_file_host(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").lower()
    return any(
        host == suffix.lstrip(".") or host.endswith(suffix)
        for suffix in UPLOADTHING_HOST_SUFFIXES
    )


def resolve_filename(url: str, provided: str | None) -> str:
    """Prefer the explicit filename, otherwise derive one from the URL path."""
    if provided and os.path.splitext(provided)[1].lower() in ALLOWED_EXTENSIONS:
        return provided
    candidate = unquote(os.path.basename(urlparse(url).path))
    if "~" in candidate and not os.path.splitext(candidate)[1].lower():
        # UploadThing keys look like "<sqid>~<filename>.<ext>"
        after_key = candidate.split("~", 1)[1]
        if os.path.splitext(after_key)[1].lower() in ALLOWED_EXTENSIONS:
            return after_key
    if os.path.splitext(candidate)[1].lower() in ALLOWED_EXTENSIONS:
        return candidate
    return provided or "unnamed"


def download_file_from_url(url: str) -> bytes:
    with requests.get(url, stream=True, timeout=(10, 60)) as response:
        response.raise_for_status()
        content_length = response.headers.get("Content-Length")
        if content_length and content_length.isdigit() and int(content_length) > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail="File exceeds the 15 MB size limit.",
            )
        downloaded = 0
        chunks = []
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            downloaded += len(chunk)
            if downloaded > MAX_UPLOAD_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="File exceeds the 15 MB size limit.",
                )
            chunks.append(chunk)
        return b"".join(chunks)


def extract_and_index(filename: str, data: bytes, index_id: str):
    try:
        text = extract_text_from_file(filename, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to extract text from the file: {exc}",
        ) from exc

    if not text.strip():
        raise HTTPException(
            status_code=400, detail="The file does not contain extractable text."
        )

    return add_text_to_db(index_id, text)


@router.post("/upload-url", response_model=UploadTextResponse)
def upload_document_from_url(req: UploadUrlRequest):
    url = req.url.strip()
    if not is_allowed_file_host(url):
        raise HTTPException(
            status_code=400,
            detail=(
                "url must be a public UploadThing file URL "
                "(*.ufs.sh, *.utfs.io or uploadthing.com)."
            ),
        )
    if not req.index_id.strip():
        raise HTTPException(status_code=400, detail="index_id is required.")

    filename = resolve_filename(url, req.filename)
    extension = os.path.splitext(filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {extension or 'unknown'}. "
                f"Pass the original filename or use a URL ending with an allowed extension. "
                f"Allowed: {sorted(ALLOWED_EXTENSIONS)}"
            ),
        )

    try:
        data = download_file_from_url(url)
    except HTTPException:
        raise
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to download the file: {exc}",
        ) from exc

    added_chunks, added_count = extract_and_index(filename, data, req.index_id)
    return UploadTextResponse(
        status="success",
        filename=filename,
        index_id=req.index_id,
        added_chunks=added_chunks,
        added_count=added_count,
    )


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


@router.post("/query", response_model=QueryResponse)
def query_text(req: QueryRequest):
    start_time = time.time()
    answer = query_db(req.question, index_ids=req.index_ids)
    latency = time.time() - start_time
    return QueryResponse(
        latency=latency,
        answer=answer,
    )


@router.post("/search", response_model=SearchResponse)
def similarity_search(req: SearchRequest):
    start_time = time.time()
    results = similarity_search_chunks(req.query, k=req.top_k, index_ids=req.index_ids)
    chunks = [
        ChunkSimilarityItem(content=content, score=score) for content, score in results
    ]
    latency = time.time() - start_time
    return SearchResponse(
        latency=latency,
        total_count=len(chunks),
        chunks=chunks,
    )
