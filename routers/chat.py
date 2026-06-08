from fastapi import APIRouter

chat_router = APIRouter(prefix="/chat", tags=["Chat"])

from api.chat import get_models, get_response
