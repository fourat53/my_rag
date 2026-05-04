import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

cloud_url = os.getenv("CLOUD_OLLAMA_URL", "")
cloud_key = os.getenv("CLOUD_OLLAMA_API_KEY", "")


def get_llm(model: str):
    return ChatOpenAI(
        model=model,
        temperature=0.3,
        base_url=cloud_url,
        api_key=cloud_key,
    )


def query_db_stream(question: str, model: str):
    llm = get_llm(model)
    prompt = ChatPromptTemplate.from_messages([("user", "{question}")])
    chain = prompt | llm
    for chunk in chain.stream({"question": question}):
        if chunk.content:
            yield chunk.content
