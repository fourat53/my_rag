from fastapi import APIRouter

documents_router = APIRouter(prefix="/documents", tags=["Documents"])

from api.documents import load_file, process_text, split_text
