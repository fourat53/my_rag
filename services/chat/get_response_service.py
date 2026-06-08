from models import GeminiResponseError, get_checked_env_var
from langchain_google_genai import ChatGoogleGenerativeAI
import logging, os

logger = logging.getLogger(__name__)


def get_gemini_response(
    query: str,
    model: str,
    temperature: float,
) -> str:
    api_key = get_checked_env_var("GEMINI_API_KEY", logger)

    try:
        llm = ChatGoogleGenerativeAI(
            model=f"models/{model}",
            temperature=temperature,
            google_api_key=api_key,
        )

        response = llm.invoke(query)

        return response.model_dump()

    except Exception as e:
        msg = f"❌ Failed generating Gemini response using model {model}: {str(e)}"
        logger.exception(msg)
        raise GeminiResponseError(msg) from e
