from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai
from typing import List, Dict
import json, os

api_key = os.getenv("GEMINI_API_KEY")


class ChatGemini:
    @staticmethod
    def get_all_models() -> List[Dict[str, str]]:
        client = genai.Client(api_key=api_key)
        usable_models = []

        try:
            for m in client.models.list():
                if m.supported_actions and "generateContent" in m.supported_actions:
                    m_dict = m.model_dump() if hasattr(m, "model_dump") else vars(m)
                    clean_model = {
                        k: v
                        for k, v in m_dict.items()
                        if v is not None
                        and k != "tuned_model_info"
                        and k != "supported_actions"
                    }
                    usable_models.append(clean_model)
        except Exception as e:
            return [{"error": f"Failed to list models: {str(e)}"}]

        return usable_models

    @staticmethod
    def chat_with_gemini(
        query: str, model: str = "models/gemini-flash-latest", temperature: float = 0
    ) -> str:
        llm = ChatGoogleGenerativeAI(
            model=model, temperature=temperature, google_api_key=api_key
        )

        response = llm.invoke(query)

        return json.dumps(response.model_dump(), indent=2)
