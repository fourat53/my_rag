from fastapi import APIRouter
from models.rag_model import (
    AddTextRequest,
    DeleteTextRequest,
    QueryRequest,
    BasicResponse,
    QueryResponse,
)
from services.rag_service import add_text_to_db, delete_text_from_db, query_db

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/add", response_model=BasicResponse)
def add_text(req: AddTextRequest):
    add_text_to_db(req.index_id, req.text)
    return BasicResponse(status="success")


@router.delete("/delete", response_model=BasicResponse)
def delete_text(req: DeleteTextRequest):
    delete_text_from_db(req.index_id)
    return BasicResponse(status="success")


@router.post("/query", response_model=QueryResponse)
def query_text(req: QueryRequest):
    answer = query_db(req.question)
    return QueryResponse(answer=answer)
