from fastapi import APIRouter

main_router = APIRouter()

# Chat Router
from api.chat import get_response, get_models

chat_router = APIRouter(prefix="/chat", tags=["Chat Operations"])
chat_router.add_api_route("/get_response", get_response.get_response, methods=["POST"])
chat_router.add_api_route("/models", get_models.get_models, methods=["GET"])
main_router.include_router(chat_router)

# Documents Router
from api.documents import load

documents_router = APIRouter(prefix="/documents", tags=["Documents Operations"])
documents_router.add_api_route("/load", load.load, methods=["POST"])
main_router.include_router(documents_router)
