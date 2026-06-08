from models import ModelsRequest, ModelsResponse, GeminiModelsError, get_checked_env_var
from services.chat import get_gemini_models
from fastapi import HTTPException, status
from routers.chat import chat_router


@chat_router.post(
    "/get_models",
    response_model=ModelsResponse,
)
def get_models(req: ModelsRequest):
    try:
        models = get_gemini_models(req.with_details)

        return {
            "status": "success",
            "models_count": len(models),
            "models": models,
        }

    except GeminiModelsError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"❌ Internal server error: {str(e)}",
        )
