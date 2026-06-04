from services.chat.gemini import ChatGemini
from models.request import ChatRequest
from models.response import ChatResponse
from .router import chat_router
import json


@chat_router.post("/get_response", response_model=ChatResponse)
async def get_response(req: ChatRequest):
    try:
        response_json = ChatGemini.chat_with_gemini(
            query=req.query, model=req.model, temperature=req.temperature
        )
        return {"status": "success", "response": json.loads(response_json)}
    except Exception as e:
        return {"status": "error", "message": str(e)}
