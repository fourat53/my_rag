import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

gemini_key = os.getenv("GEMINI_API_KEY", "")


def get_llm(model: str):
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0.3,
        google_api_key=gemini_key,
    )


def query_db_stream(question: str, model: str):
    llm = get_llm(model)
    prompt = ChatPromptTemplate.from_messages([("user", "{question}")])
    chain = prompt | llm
    for chunk in chain.stream({"question": question}):
        if chunk.content:
            yield chunk.content
