import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage


def get_llm(model: str, base_url: str):
    return ChatOpenAI(
        model=model,
        temperature=0.3,
        base_url=base_url,
    )


def query_stream(
    question: str,
    model: str,
    base_url: str | None = None,
    history: list[dict] | None = None,
    api_key: str | None = None,
):
    llm = get_llm(model, base_url)
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
