from models import ChatRequest, ChatResponse, GeminiResponseError
from services.chat import get_gemini_response
from fastapi import HTTPException, status
from routers.chat import chat_router


@chat_router.post(
    "/get_response",
    response_model=ChatResponse,
)
def get_response(req: ChatRequest):
    try:
        response = get_gemini_response(
            query=req.query,
            model=req.model,
            temperature=req.temperature,
        )

        return {
            "status": "success",
            "response": response,
        }

    except GeminiResponseError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"❌ Internal server error: {str(e)}",
        )
