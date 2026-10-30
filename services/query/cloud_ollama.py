import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage


def get_llm(model: str, base_url: str, api_key: str | None = None):
    return ChatOpenAI(
        model=model,
        temperature=0.3,
        api_key=api_key or os.getenv("OLLAMA_API_KEY"),
        base_url=base_url,
    )


def query_stream(
    question: str,
    model: str,
    base_url: str,
    history: list[dict] | None = None,
    api_key: str | None = None,
):
    llm = get_llm(model, base_url, api_key)
    messages = [
        (
            HumanMessage(content=m["content"])
            if m.get("role") == "user"
            else AIMessage(content=m["content"])
        )
        for m in (history or [])
        if m.get("content")
    ]
    messages.append(HumanMessage(content=question))
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content
