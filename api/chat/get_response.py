from api.models.request import ChatRequest
from chat.gemini import ChatGemini
from api import chat_router


@chat_router.post("/get_response")
async def get_response(req: ChatRequest):
    try:
        gemini = ChatGemini(model=req.model, temperature=req.temperature)
        response = gemini.chat_with_gemini(query=req.query)
        return {"status": "success", "response": response}
    except Exception as e:
        return {"status": "error", "message": str(e)}
