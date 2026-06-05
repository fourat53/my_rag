from services.chat.gemini import ChatGemini
from models.request import ChatRequest
import json


async def get_response(req: ChatRequest):
    try:
        response_json = ChatGemini.chat_with_gemini(
            query=req.query, model=req.model, temperature=req.temperature
        )
        return {"status": "success", "response": json.loads(response_json)}
    except Exception as e:
        return {"status": "error", "message": str(e)}
