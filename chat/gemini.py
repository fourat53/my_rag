from langchain_google_genai import ChatGoogleGenerativeAI
import json


class ChatGemini:
    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        temperature: float = 0,
    ):
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)

    def chat_with_gemini(self, query: str):
        response_gemini = self.llm.invoke(query)
        return json.dumps(response_gemini, indent=2)
