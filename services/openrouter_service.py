import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

openrouter_key = os.getenv("OPENROUTER_API_KEY", "")


def get_llm(model: str):
    return ChatOpenAI(
        model=model,
        temperature=0.3,
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_key,
    )


def query_db_stream(question: str, model: str):
    llm = get_llm(model)
    prompt = ChatPromptTemplate.from_messages([("user", "{question}")])
    chain = prompt | llm
    for chunk in chain.stream({"question": question}):
        if chunk.content:
            yield chunk.content
