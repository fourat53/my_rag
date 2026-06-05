from services.chat.gemini import ChatGemini


async def get_models():
    models = ChatGemini.get_all_models()
    return {"status": "success", "models_count": len(models), "models": models}
