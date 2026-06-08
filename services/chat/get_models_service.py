from models import GeminiModelsError, get_checked_env_var
from typing import List, Dict
from google import genai
import logging, os

logger = logging.getLogger(__name__)


def get_gemini_models(with_details: bool) -> List[Dict]:
    api_key = get_checked_env_var("GEMINI_API_KEY", logger)

    try:
        client = genai.Client(api_key=api_key)

        usable_models = []

        for model in client.models.list():
            if model.supported_actions and "generateContent" in model.supported_actions:
                model_dict = (
                    model.model_dump() if hasattr(model, "model_dump") else vars(model)
                )

                clean_model = {
                    k: v
                    for k, v in model_dict.items()
                    if v is not None
                    and k != "tuned_model_info"
                    and k != "supported_actions"
                }

                if with_details:
                    usable_models.append(clean_model)
                else:
                    usable_models.append(clean_model["name"].split("/")[-1])

        return usable_models

    except Exception as e:
        msg = f"❌ Failed to retrieve Gemini models: {str(e)}"
        logger.exception(msg)
        raise GeminiModelsError(msg) from e
