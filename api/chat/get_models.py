from services.chat.gemini import ChatGemini
from models.response import ModelsResponse
from .router import chat_router


@chat_router.get("/models", response_model=ModelsResponse)
async def list_models():
    models = ChatGemini.get_all_models()
    return {"status": "success", "models_count": len(models), "models": models}
